[CmdletBinding()]
param(
    [ValidateSet('List', 'Preflight', 'Hermite', 'Heavy', 'Postprocess', 'Audit', 'All')]
    [string]$Phase = 'List',
    [string]$Python = 'python',
    [switch]$Submission,
    # -Phase Hermite only.  P0 = 10 is now part of the default production set:
    # the real, cache-blocked, threaded kernel of src\mqcle\hermite\kernel.py
    # brought it from 300-500 h to a run that is worth making, and it is the
    # regime where the methods differ most.  -IncludeP10 is kept so that an
    # existing command line still parses; it no longer changes anything.
    # -SkipP10 is the switch that now omits it.
    [switch]$IncludeP10,
    [switch]$SkipP10,
    # Worker threads for the Hermite kernel.  0 means os.cpu_count().
    [int]$Threads = 0,
    # Arnoldi subspace dimension for the production propagator.  Cost per
    # substep is KrylovM operator applications; storage is KrylovM + 1 state
    # vectors, which is what bounds it at P0 = 10.  0 keeps the script default.
    [int]$KrylovM = 0,
    # Requested local error per Arnoldi substep.  0 keeps the script default of
    # 1e-10.  Loosening this to 1e-8 roughly halves the cost and raises the
    # physical-frame residual floor by the same factor, because that diagnostic
    # tracks the integrator tolerance rather than the operator.
    [double]$KrylovTol = 0
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$RawDir = Join-Path $ProjectRoot 'results\raw'
$LogDir = Join-Path $ProjectRoot 'results\logs'
$Manifest = Join-Path $ProjectRoot 'MANIFEST.sha256'
Set-Location -LiteralPath $ProjectRoot
New-Item -ItemType Directory -Path $RawDir, $LogDir -Force | Out-Null

# Compiled bytecode is written by the very Python children this runner starts,
# so it must never be part of the revision baseline below, and there is no
# reason to leave it in a production tree at all.  Setting this before any
# child starts removes the cause rather than only the symptom.
$env:PYTHONDONTWRITEBYTECODE = '1'

# A Heavy run can last for days.  PowerShell loads this file once, whereas each
# Python child imports the then-current files from disk.  Refuse to mix code or
# configuration revisions if the workspace changes while a run is active.
#
# The selection is deliberately explicit.  `Get-ChildItem -LiteralPath <dir>
# -Recurse -Include '*.py'` does NOT filter: with -LiteralPath the -Include
# pattern is matched against the leaf of the path argument, which is a
# directory name, and every recursed file is returned instead.  That silently
# admitted `__pycache__\*.pyc` to the baseline -- files CPython rewrites
# whenever an imported module's cached bytecode is stale, which the runner's
# own children do.  The guard then fired on a change it had itself caused.
# Filtering on the extension with Where-Object has no such provider quirk.
$PipelineCodeFiles = @(
    Get-Item -LiteralPath $PSCommandPath
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'src') -File -Recurse |
        Where-Object { $_.Extension -eq '.py' }
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'scripts') -File -Recurse |
        Where-Object { $_.Extension -eq '.py' }
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'configs') -File -Recurse |
        Where-Object { $_.Extension -in '.yaml', '.yml' }
) | Where-Object {
        $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne '.pyc'
    } | Sort-Object FullName -Unique

# A selection that silently returns the wrong set is what caused the failure
# this replaces, so assert the set instead of assuming it.
if ($PipelineCodeFiles.Count -lt 20) {
    throw ("Pipeline revision baseline looks wrong: only " +
           "$($PipelineCodeFiles.Count) source files were found under src, " +
           'scripts and configs.  Refusing to run with an unreliable guard.')
}
$baselineOffenders = $PipelineCodeFiles |
    Where-Object { $_.Extension -notin '.py', '.yaml', '.yml', '.ps1' }
if ($baselineOffenders) {
    throw ('Pipeline revision baseline contains generated files: ' +
           (($baselineOffenders | ForEach-Object { $_.Name }) -join ', '))
}

$PipelineCodeHashes = @{}
foreach ($file in $PipelineCodeFiles) {
    $PipelineCodeHashes[$file.FullName] =
        (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
}

function Assert-PipelineCodeUnchanged {
    $changed = @()
    foreach ($path in $PipelineCodeHashes.Keys) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $changed += $path
            continue
        }
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($hash -ne $PipelineCodeHashes[$path]) { $changed += $path }
    }
    if ($changed.Count -gt 0) {
        $relative = $changed | ForEach-Object {
            if ($_.StartsWith($ProjectRoot)) {
                $_.Substring($ProjectRoot.Length + 1)
            }
            else { $_ }
        }
        throw ("Pipeline code/configuration changed after this process started. " +
               "Completed outputs are preserved. Restart the same phase so one " +
               "consistent revision is used. Changed: " + ($relative -join ', '))
    }
}

function Test-JsonAllPassed([string]$Path) {
    try {
        $doc = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return ($doc.all_passed -eq $true)
    }
    catch {
        return $false
    }
}

function Assert-Completed([string]$RelativePath) {
    $path = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required completed result is missing: $RelativePath"
    }
    if ($path.EndsWith('.json') -and -not (Test-JsonAllPassed $path)) {
        throw "Required result does not record all_passed=true: $RelativePath"
    }
}

function Assert-P10Accepted {
    Assert-Completed 'results\raw\07c_p10_refinement_acceptance.json'
    $inputs = [ordered]@{
        fine_json = 'results\raw\07_coefficient_propagation_P10.json'
        fine_h5 = 'results\raw\07_coefficient_propagation_P10.h5'
        coarse_json = 'results\failed\P10_2048x1024_20260829\07_coefficient_propagation_P10.json'
        coarse_h5 = 'results\failed\P10_2048x1024_20260829\07_coefficient_propagation_P10.h5'
    }
    $auditPath = Join-Path $ProjectRoot 'results\raw\07c_p10_refinement_acceptance.json'
    $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    foreach ($name in $inputs.Keys) {
        $path = $inputs[$name]
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $path) -PathType Leaf)) {
            throw "Required P10 refinement input is missing: $path"
        }
        $property = $audit.results.input_sha256.PSObject.Properties[$name]
        if ($null -eq $property) {
            throw "P10 refinement audit does not record the $name input hash"
        }
        $actual = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot $path) `
            -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne ([string]$property.Value).ToLowerInvariant()) {
            throw "P10 refinement audit is stale for $path; rerun scripts\07c_p10_refinement_acceptance.py"
        }
    }
}

function Assert-JsonGrid {
    param(
        [Parameter(Mandatory)][string]$RelativePath,
        [Parameter(Mandatory)][int]$ExpectedNR,
        [Parameter(Mandatory)][int]$ExpectedNP,
        [Parameter(Mandatory)][double]$ExpectedRMin,
        [Parameter(Mandatory)][double]$ExpectedRMax,
        [Parameter(Mandatory)][double]$ExpectedPMin,
        [Parameter(Mandatory)][double]$ExpectedPMax
    )
    $path = Join-Path $ProjectRoot $RelativePath
    $doc = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    $config = $doc.metadata.config
    $gridProperty = $config.PSObject.Properties['grid']
    if ($null -ne $gridProperty) {
        $grid = $gridProperty.Value
    }
    else {
        $grid = $doc.results.coefficient.grid
    }
    if ($grid.n_R -ne $ExpectedNR -or $grid.n_P -ne $ExpectedNP -or
        $grid.R_min -ne $ExpectedRMin -or $grid.R_max -ne $ExpectedRMax -or
        $grid.P_min -ne $ExpectedPMin -or $grid.P_max -ne $ExpectedPMax) {
        throw "Unexpected production grid in ${RelativePath}: " +
              "$($grid.n_R)x$($grid.n_P), " +
              "R=[$($grid.R_min),$($grid.R_max)], " +
              "P=[$($grid.P_min),$($grid.P_max)]"
    }
}

function Invoke-NativeLogged {
    # Runs a native command, mirrors its merged stdout/stderr to the console
    # and to $LogPath, and returns the process exit code.
    #
    # Windows PowerShell 5.1 wraps every stderr line of a native command in an
    # ErrorRecord once stderr is merged with 2>&1.  Under the script-wide
    # $ErrorActionPreference = 'Stop' the first such line -- a Python
    # SyntaxWarning or DeprecationWarning, a matplotlib UserWarning, a progress
    # message -- raised NativeCommandError and aborted the stage while the
    # child was still running and would have exited 0.  Inside this function
    # stderr is ordinary log text, and success is decided by the exit code
    # alone.
    param(
        [Parameter(Mandatory)][string]$Command,
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Arguments,
        [Parameter(Mandatory)][string]$LogPath
    )
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    # A command that cannot be started leaves $LASTEXITCODE untouched; the
    # sentinel makes that a failure instead of inheriting an earlier 0.
    $global:LASTEXITCODE = -1
    try {
        & $Command @Arguments 2>&1 |
            ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() }
                else { $_ }
            } |
            Tee-Object -FilePath $LogPath |
            Out-Host
        $exitCode = $global:LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    return $exitCode
}

function Invoke-PythonStage {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Output,
        [Parameter(Mandatory)][string]$Log,
        [Parameter(Mandatory)][string[]]$StageArguments,
        [switch]$OutputHasNoChecks
    )
    Assert-PipelineCodeUnchanged
    $outputPath = Join-Path $ProjectRoot $Output
    $logPath = Join-Path $ProjectRoot $Log
    if (Test-Path -LiteralPath $outputPath -PathType Leaf) {
        if ($OutputHasNoChecks -or (Test-JsonAllPassed $outputPath)) {
            Write-Host "[skip] $Name -- validated output already exists"
            return
        }
        throw "$Name has an existing output that is not valid. Preserve or move it before rerunning: $Output"
    }
    Write-Host "`n[start] $Name"
    $exitCode = Invoke-NativeLogged $Python (@('-u') + $StageArguments) $logPath
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $Log"
    }
    if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
        throw "$Name exited successfully but did not create $Output"
    }
    if (-not $OutputHasNoChecks -and -not (Test-JsonAllPassed $outputPath)) {
        throw "$Name created $Output but one or more recorded checks failed"
    }
    Write-Host "[done] $Name"
}

function Invoke-PythonChecked {
    param([string]$Name, [string[]]$StageArguments, [string]$Log)
    Assert-PipelineCodeUnchanged
    $logPath = Join-Path $ProjectRoot $Log
    Write-Host "`n[start] $Name"
    $exitCode = Invoke-NativeLogged $Python $StageArguments $logPath
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $Log"
    }
    Write-Host "[done] $Name"
}

function Invoke-NativeChecked {
    param([string]$Name, [string]$Command, [string[]]$Arguments, [string]$Log)
    Assert-PipelineCodeUnchanged
    $logPath = Join-Path $ProjectRoot $Log
    Write-Host "`n[start] $Name"
    $exitCode = Invoke-NativeLogged $Command $Arguments $logPath
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $Log"
    }
    Write-Host "[done] $Name"
}

function Invoke-Preflight {
    foreach ($path in @(
        'configs\tully_default.yaml',
        'scripts\07_coefficient_propagation.py',
        'scripts\07g_gh_coefficient_propagation.py',
        'scripts\07cg_gh_basis_covariance.py',
        'scripts\07k_gh_closure_validation.py',
        'scripts\10g_gh_crosscheck.py',
        'src\mqcle\hermite\covariance.py',
        'scripts\09g_gh_convergence.py',
        'src\mqcle\hermite\operator.py',
        'src\mqcle\hermite\frame.py',
        'src\mqcle\hermite\propagator.py',
        'src\mqcle\hermite\kernel.py',
        'tests\test_gaussian_hermite.py',
        'scripts\07c_p10_refinement_acceptance.py',
        'scripts\08_leakage_diagnostic.py',
        'scripts\10_schrodinger_convergence.py',
        'scripts\10b_qcle_grid_crosscheck.py',
        'scripts\11_pbme_projected.py',
        'scripts\11b_pbme_statistics.py',
        'scripts\18_pbme_phase_space.py',
        'scripts\19_phase_space_comparison.py',
        'scripts\20_focused_pbme_observables.py',
        'scripts\13_acceptance.py'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $path))) {
            throw "Pipeline file missing: $path"
        }
    }
    & $Python -c "import h5py, matplotlib, numpy, pandas, scipy, sympy, yaml; print('Python scientific runtime: OK')"
    if ($LASTEXITCODE -ne 0) {
        throw 'Python dependency preflight failed. Run: python -m pip install --user -r .\requirements.txt'
    }
    Assert-Completed 'results\raw\11_pbme_projected_P40.json'
    Assert-Completed 'results\raw\11b_pbme_seed_convergence_P40.json'
    Assert-Completed 'results\raw\11c_pbme_timestep_convergence.json'
    Write-Host 'Completed P40 projected-PBME and time-step results: OK'
}

function Assert-AllProductionInputs {
    $checked = @(
        'results\raw\07c_p10_refinement_acceptance.json',
        'results\raw\13_P20_wide_momentum_box.json',
        'results\raw\13_P20_wide_momentum_box.h5',
        'results\raw\07_coefficient_propagation_P30.json',
        'results\raw\07_coefficient_propagation_P30.h5',
        'results\raw\07_coefficient_propagation_P40.json',
        'results\raw\07_coefficient_propagation_P40.h5',
        'results\raw\07_coefficient_propagation_P70.json',
        'results\raw\07_coefficient_propagation_P70.h5',
        'results\raw\07_coefficient_propagation_P100.json',
        'results\raw\07_coefficient_propagation_P100.h5',
        'results\raw\08_leakage_P10.json',
        'results\raw\08_leakage_P20.json',
        'results\raw\08_leakage_P30.json',
        'results\raw\08_leakage_P40.json',
        'results\raw\08_leakage_P70.json',
        'results\raw\08_leakage_P100.json',
        'results\raw\10_schrodinger_convergence_P10.json',
        'results\raw\10_schrodinger_convergence_P20.json',
        'results\raw\10_schrodinger_convergence_P30.json',
        'results\raw\10_schrodinger_convergence_P40.json',
        'results\raw\10_schrodinger_convergence_P70.json',
        'results\raw\10_schrodinger_convergence_P100.json',
        'results\raw\10b_qcle_grid_crosscheck_P40.json',
        'results\raw\10b_qcle_grid_crosscheck_P40.h5',
        'results\raw\11_pbme_projected_P10.json',
        'results\raw\11_pbme_projected_P20.json',
        'results\raw\11_pbme_projected_P30.json',
        'results\raw\11_pbme_projected_P40.json',
        'results\raw\11_pbme_projected_P70.json',
        'results\raw\11_pbme_projected_P100.json',
        'results\raw\11b_pbme_seed_convergence_P10.json',
        'results\raw\11b_pbme_seed_convergence_P20.json',
        'results\raw\11b_pbme_seed_convergence_P30.json',
        'results\raw\11b_pbme_seed_convergence_P40.json',
        'results\raw\11b_pbme_seed_convergence_P70.json',
        'results\raw\11b_pbme_seed_convergence_P100.json',
        'results\raw\11c_pbme_timestep_convergence.json',
        'results\raw\11_pbme_projected_P40_C0.005.json',
        'results\raw\11_pbme_projected_P40_C0.0075.json',
        'results\raw\11_pbme_projected_P40_C0.03.json',
        'results\raw\11_pbme_projected_P40_C0.06.json',
        'results\raw\11_pbme_projected_P40_phi.json',
        'results\raw\18_pbme_phase_space_P20.json',
        'results\raw\18_pbme_phase_space_P30.json',
        'results\raw\18_pbme_phase_space_P70.json',
        'results\raw\18_pbme_phase_space_P100.json',
        'results\raw\19_phase_space_comparison_P20.json',
        'results\raw\19_phase_space_comparison_P30.json',
        'results\raw\19_phase_space_comparison_P70.json',
        'results\raw\19_phase_space_comparison_P100.json',
        'results\raw\20_focused_pbme_observables_P10.json',
        'results\raw\20_focused_pbme_observables_P20.json',
        'results\raw\20_focused_pbme_observables_P30.json',
        'results\raw\20_focused_pbme_observables_P40.json',
        'results\raw\20_focused_pbme_observables_P70.json',
        'results\raw\20_focused_pbme_observables_P100.json'
    )
    foreach ($path in $checked) { Assert-Completed $path }
    Assert-P10Accepted
    Assert-JsonGrid 'results\raw\13_P20_wide_momentum_box.json' `
        4096 1024 -45 45 -20 60
    Assert-JsonGrid 'results\raw\07_coefficient_propagation_P10.json' `
        4096 2048 -45 45 -30 50
    Assert-JsonGrid 'results\raw\07_coefficient_propagation_P30.json' `
        1024 512 -45 45 8 52
    Assert-JsonGrid 'results\raw\07_coefficient_propagation_P40.json' `
        1024 512 -45 45 18 62
    Assert-JsonGrid 'results\raw\07_coefficient_propagation_P70.json' `
        2048 1024 -35 35 38 102
    Assert-JsonGrid 'results\raw\07_coefficient_propagation_P100.json' `
        2048 1024 -35 35 60 140
    Assert-JsonGrid 'results\raw\08_leakage_P20.json' `
        4096 1024 -45 45 -20 60
    Assert-JsonGrid 'results\raw\08_leakage_P10.json' `
        4096 2048 -45 45 -30 50
    Assert-JsonGrid 'results\raw\08_leakage_P30.json' `
        1024 512 -45 45 8 52
    Assert-JsonGrid 'results\raw\08_leakage_P70.json' `
        2048 1024 -35 35 38 102
    foreach ($path in @('results\raw\12_leakage_reanalysis.json',
                         'results\raw\12_coupling_scan_projected_pbme.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $path) -PathType Leaf)) {
            throw "Required aggregate is missing: $path"
        }
    }
    $aggregate = Get-Content -LiteralPath (Join-Path $RawDir '12_leakage_reanalysis.json') `
        -Raw | ConvertFrom-Json
    if ($aggregate.results.n_regimes -lt 6 -or
        $aggregate.results.response_variable -notmatch 'projected') {
        throw 'Stage 12 aggregate does not contain all six projected-response regimes'
    }
    Write-Host 'All final production inputs: OK'
}

function Assert-AggregateInputs {
    $required = @(
        'results\raw\07c_p10_refinement_acceptance.json',
        'results\raw\13_P20_wide_momentum_box.json',
        'results\raw\07_coefficient_propagation_P30.json',
        'results\raw\07_coefficient_propagation_P40.json',
        'results\raw\07_coefficient_propagation_P70.json',
        'results\raw\07_coefficient_propagation_P100.json',
        'results\raw\08_leakage_P10.json',
        'results\raw\08_leakage_P20.json',
        'results\raw\08_leakage_P30.json',
        'results\raw\08_leakage_P40.json',
        'results\raw\08_leakage_P70.json',
        'results\raw\08_leakage_P100.json',
        'results\raw\11_pbme_projected_P10.json',
        'results\raw\11_pbme_projected_P20.json',
        'results\raw\11_pbme_projected_P30.json',
        'results\raw\11_pbme_projected_P40.json',
        'results\raw\11_pbme_projected_P70.json',
        'results\raw\11_pbme_projected_P100.json',
        'results\raw\11_pbme_projected_P40_C0.005.json',
        'results\raw\11_pbme_projected_P40_C0.0075.json',
        'results\raw\11_pbme_projected_P40_C0.03.json',
        'results\raw\11_pbme_projected_P40_C0.06.json'
    )
    foreach ($path in $required) { Assert-Completed $path }
    Assert-P10Accepted
    Write-Host 'All six-regime leakage-aggregate inputs: OK'
}

function Invoke-Hermite {
    # The primary branch: the Gaussian-Hermite coefficient system of
    # Eq. (finite-Hermite), propagated directly.  It is placed before the
    # reduced-field runs because it is what the theorem describes; the runs
    # below it produce the reference it is checked against.
    #
    # This phase is separable on purpose.  The branch is 3.2x to 8.1x the cost
    # of the reference it replaces and its P0=10 regime is by far the longest,
    # so it must be startable, resumable and stoppable without touching the
    # rest of the pipeline.  Every stage below is skipped if its validated
    # output already exists.
    Invoke-Preflight

    Invoke-PythonChecked 'Gaussian-Hermite gates GH-01..GH-15' @(
        '-m', 'pytest', '.\tests\test_gaussian_hermite.py', '-q') `
        'results\logs\gh_gates.log'

    # Gate GH-11 at production scale.  Minutes, and it belongs here rather
    # than after the regimes: if the operator is not covariant under a change
    # of subsystem basis then the closure is an artefact of the basis it was
    # written in, and no amount of propagation afterwards would be worth
    # having.  Cheap checks that can invalidate expensive ones run first.
    Invoke-PythonStage 'Gaussian-Hermite basis covariance (GH-11)' `
        'results\raw\07cg_gh_basis_covariance.json' `
        'results\logs\07cg_covariance.log' @(
        '.\scripts\07cg_gh_basis_covariance.py')

    # Gate GH-10 at production scale: the Hermite branch against the Strang
    # reference on a sequence of grids.  The check is not the size of the
    # difference but that it FALLS under refinement -- two discretisations of
    # one equation converging to each other.  Minutes.
    Invoke-PythonStage 'Gaussian-Hermite cross-check vs the reference (GH-10)' `
        'results\raw\07g_gh_crosscheck_P40_convergence.json' `
        'results\logs\10g_crosscheck.log' @(
        '.\scripts\10g_gh_crosscheck.py')

    Invoke-PythonStage 'Gaussian-Hermite convergence study (P40)' `
        'results\raw\09g_gh_convergence_P40.json' 'results\logs\09g_conv.log' @(
        '.\scripts\09g_gh_convergence.py')

    # The closure theorem, tested in bases that contain the sectors it forbids.
    # A K=2 basis cannot test closure and an even basis cannot test parity, so
    # this runs the full even sector through degree four and a mixed-parity
    # basis, each with a control that starts inside the forbidden sector.  It
    # is a validation gate and runs before the production regimes: if the
    # operator did populate order four, no production number would be worth
    # having.  Minutes.
    $kargs = @('.\scripts\07k_gh_closure_validation.py')
    if ($Threads -gt 0) { $kargs += @('--threads', "$Threads") }
    if ($KrylovM -gt 0) { $kargs += @('--krylov-m', "$KrylovM") }
    Invoke-PythonStage 'Gaussian-Hermite closure and parity validation' `
        'results\raw\07k_gh_closure_validation.json' `
        'results\logs\07k_closure.log' $kargs

    # Cheapest first, so that a machine-level problem surfaces in minutes
    # rather than after a day.  P0 = 10 is last because it is the longest, and
    # it is included by default: it is the regime in which the methods differ
    # most, so a comparison paper that omits it is not making its own point.
    $Momenta = @(40, 30, 100, 70, 20, 10)
    if ($SkipP10) { $Momenta = @(40, 30, 100, 70, 20) }
    foreach ($P0 in $Momenta) {
        $gargs = @('.\scripts\07g_gh_coefficient_propagation.py',
                   '--P0', "$P0", '--save-field')
        if ($Threads -gt 0) { $gargs += @('--threads', "$Threads") }
        if ($KrylovM -gt 0) { $gargs += @('--krylov-m', "$KrylovM") }
        if ($KrylovTol -gt 0) { $gargs += @('--krylov-tol', "$KrylovTol") }
        Invoke-PythonStage "Gaussian-Hermite coefficient propagation P$P0" `
            "results\raw\07g_gh_coefficient_propagation_P$P0.json" `
            "results\logs\07g_P$P0.log" $gargs
    }

    if ($SkipP10) {
        Write-Host ""
        Write-Host "[note] P0=10 was skipped by -SkipP10.  A regime with no Hermite artifact"
        Write-Host "       emits \ghAbsent and is dropped from the figures rather than filled"
        Write-Host "       in from the branch it replaces, so the paper will say so in print."
    }
    if ($IncludeP10) {
        Write-Host ""
        Write-Host "[note] -IncludeP10 is obsolete: P0=10 is now in the default set."
    }

    # End with a rebuilt paper rather than a directory of JSON.  This reuses
    # Invoke-Postprocess rather than repeating it: the macro generator reads
    # the .aux for cross-references, so macros and LaTeX passes have to be
    # interleaved in a specific order, and a shortened `latexmk` here would
    # silently produce a document whose equation numbers in the supplement
    # cite the previous build.
    Assert-AllProductionInputs
    Invoke-Postprocess
    Invoke-Audit

    Write-Host ""
    Write-Host "[done] Hermite branch complete."
    Write-Host "       results\logs\14_format_audit.log  -- expect '0 mandatory failures'"
    Write-Host "       results\logs\13_acceptance.log"
    Write-Host "       manuscript\mqcle_hermite_closure.pdf"
}

function Invoke-Heavy {
    Invoke-Preflight

    $p10Output = 'results\raw\07_coefficient_propagation_P10.json'
    $p10OutputPath = Join-Path $ProjectRoot $p10Output
    if (Test-Path -LiteralPath $p10OutputPath -PathType Leaf) {
        Write-Host '[skip] P10 coefficient/QCLE/quantum/focused benchmark -- completed raw output exists'
    }
    else {
        Assert-PipelineCodeUnchanged
        Write-Host "`n[start] P10 coefficient/QCLE/quantum/focused benchmark"
        $p10ExitCode = Invoke-NativeLogged $Python @(
            '-u', '.\scripts\07_coefficient_propagation.py',
            '--config', '.\configs\tully_default.yaml',
            '--P0', '10', '--P-center', '10', '--P-half', '40',
            '--n-R', '4096', '--n-P', '2048',
            '--n-save', '20', '--n-samples', '200000',
            '--out', '.\results\raw\07_coefficient_propagation_P10.json'
        ) (Join-Path $ProjectRoot 'results\logs\07_P10.log')
        if (-not (Test-Path -LiteralPath $p10OutputPath -PathType Leaf)) {
            throw "P10 coefficient benchmark exited with code $p10ExitCode and did not create $p10Output"
        }
        Write-Host "[provisional] P10 coefficient benchmark exited with code $p10ExitCode; validating the saved two-grid observable evidence"
    }
    Assert-JsonGrid 'results\raw\07_coefficient_propagation_P10.json' `
        4096 2048 -45 45 -30 50
    Invoke-PythonChecked 'P10 simultaneous two-axis refinement acceptance' @(
        '.\scripts\07c_p10_refinement_acceptance.py',
        '--fine-json', '.\results\raw\07_coefficient_propagation_P10.json',
        '--fine-h5', '.\results\raw\07_coefficient_propagation_P10.h5',
        '--coarse-json', '.\results\failed\P10_2048x1024_20260829\07_coefficient_propagation_P10.json',
        '--coarse-h5', '.\results\failed\P10_2048x1024_20260829\07_coefficient_propagation_P10.h5',
        '--out', '.\results\raw\07c_p10_refinement_acceptance.json'
    ) 'results\logs\07c_P10_refinement.log'
    Assert-P10Accepted

    Invoke-PythonStage 'P20 widened-box coefficient/QCLE/quantum/focused benchmark' `
        'results\raw\13_P20_wide_momentum_box.json' 'results\logs\13_P20_wide.log' @(
        '.\scripts\07_coefficient_propagation.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '20', '--P-center', '20', '--P-half', '40',
        '--n-R', '4096', '--n-P', '1024',
        '--n-save', '20', '--n-samples', '200000',
        '--out', '.\results\raw\13_P20_wide_momentum_box.json')
    Assert-JsonGrid 'results\raw\13_P20_wide_momentum_box.json' `
        4096 1024 -45 45 -20 60

    Invoke-PythonStage 'P30 coefficient/QCLE/quantum/focused benchmark' `
        'results\raw\07_coefficient_propagation_P30.json' 'results\logs\07_P30.log' @(
        '.\scripts\07_coefficient_propagation.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '30', '--P-center', '30', '--P-half', '22',
        '--n-R', '1024', '--n-P', '512',
        '--n-save', '20', '--n-samples', '200000',
        '--out', '.\results\raw\07_coefficient_propagation_P30.json')
    Assert-JsonGrid 'results\raw\07_coefficient_propagation_P30.json' `
        1024 512 -45 45 8 52

    Invoke-PythonStage 'P70 coefficient/QCLE/quantum/focused benchmark' `
        'results\raw\07_coefficient_propagation_P70.json' 'results\logs\07_P70.log' @(
        '.\scripts\07_coefficient_propagation.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '70', '--P-center', '70', '--P-half', '32',
        '--R-min', '-35', '--R-max', '35', '--n-R', '2048', '--n-P', '1024',
        '--dt', '0.5', '--T', '857.5',
        '--n-save', '20', '--n-samples', '200000',
        '--out', '.\results\raw\07_coefficient_propagation_P70.json')
    Assert-JsonGrid 'results\raw\07_coefficient_propagation_P70.json' `
        2048 1024 -35 35 38 102

    Invoke-PythonStage 'P10 leakage diagnostic' `
        'results\raw\08_leakage_P10.json' 'results\logs\08_P10.log' @(
        '.\scripts\08_leakage_diagnostic.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '10', '--P-center', '10', '--P-half', '40',
        '--n-R', '4096', '--n-P', '2048',
        '--out', '.\results\raw\08_leakage_P10.json')

    Invoke-PythonStage 'P20 widened-box leakage diagnostic' `
        'results\raw\08_leakage_P20.json' 'results\logs\08_P20_wide.log' @(
        '.\scripts\08_leakage_diagnostic.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '20', '--P-center', '20', '--P-half', '40',
        '--n-R', '4096', '--n-P', '1024',
        '--out', '.\results\raw\08_leakage_P20.json')
    Assert-JsonGrid 'results\raw\08_leakage_P20.json' `
        4096 1024 -45 45 -20 60

    Invoke-PythonStage 'P30 leakage diagnostic' `
        'results\raw\08_leakage_P30.json' 'results\logs\08_P30.log' @(
        '.\scripts\08_leakage_diagnostic.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '30', '--P-center', '30', '--P-half', '22',
        '--n-R', '1024', '--n-P', '512',
        '--out', '.\results\raw\08_leakage_P30.json')

    Invoke-PythonStage 'P70 leakage diagnostic' `
        'results\raw\08_leakage_P70.json' 'results\logs\08_P70.log' @(
        '.\scripts\08_leakage_diagnostic.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '70', '--P-center', '70', '--P-half', '32',
        '--R-min', '-35', '--R-max', '35', '--n-R', '2048', '--n-P', '1024',
        '--dt', '0.5', '--T', '857.5',
        '--out', '.\results\raw\08_leakage_P70.json')

    Invoke-PythonStage 'Full-quantum reference convergence P10' `
        'results\raw\10_schrodinger_convergence_P10.json' 'results\logs\10_se_conv_P10.log' @(
        '.\scripts\10_schrodinger_convergence.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '10', '--out', '.\results\raw\10_schrodinger_convergence_P10.json')

    Invoke-PythonStage 'Full-quantum reference convergence P20' `
        'results\raw\10_schrodinger_convergence_P20.json' 'results\logs\10_se_conv_P20.log' @(
        '.\scripts\10_schrodinger_convergence.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '20', '--out', '.\results\raw\10_schrodinger_convergence_P20.json')
    Invoke-PythonStage 'Full-quantum reference convergence P30' `
        'results\raw\10_schrodinger_convergence_P30.json' 'results\logs\10_se_conv_P30.log' @(
        '.\scripts\10_schrodinger_convergence.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '30', '--out', '.\results\raw\10_schrodinger_convergence_P30.json')
    Invoke-PythonStage 'Full-quantum reference convergence P40' `
        'results\raw\10_schrodinger_convergence_P40.json' 'results\logs\10_se_conv_P40.log' @(
        '.\scripts\10_schrodinger_convergence.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '40', '--out', '.\results\raw\10_schrodinger_convergence_P40.json')
    Invoke-PythonStage 'Full-quantum reference convergence P70' `
        'results\raw\10_schrodinger_convergence_P70.json' 'results\logs\10_se_conv_P70.log' @(
        '.\scripts\10_schrodinger_convergence.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '70', '--out', '.\results\raw\10_schrodinger_convergence_P70.json')
    Invoke-PythonStage 'Full-quantum reference convergence P100' `
        'results\raw\10_schrodinger_convergence_P100.json' 'results\logs\10_se_conv_P100.log' @(
        '.\scripts\10_schrodinger_convergence.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '100', '--out', '.\results\raw\10_schrodinger_convergence_P100.json')

    Invoke-PythonStage 'Independent grid-QCLE cross-check P40' `
        'results\raw\10b_qcle_grid_crosscheck_P40.json' `
        'results\logs\10b_qcle_grid_crosscheck_P40.log' @(
        '.\scripts\10b_qcle_grid_crosscheck.py',
        '--config', '.\configs\tully_default.yaml', '--P0', '40',
        '--archive', '.\results\raw\07_coefficient_propagation_P40.json',
        '--out', '.\results\raw\10b_qcle_grid_crosscheck_P40.json')

    Invoke-PythonStage 'Projected-density PBME P10' `
        'results\raw\11_pbme_projected_P10.json' 'results\logs\11_P10.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '10', '--n-samples', '1200000', '--chunk', '200000',
        '--n-seeds', '6', '--blocks-per-chunk', '40', '--focused-samples', '200000',
        '--out', '.\results\raw\11_pbme_projected_P10.json')

    Invoke-PythonStage 'Projected-density PBME P20' `
        'results\raw\11_pbme_projected_P20.json' 'results\logs\11_P20.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '20', '--n-samples', '1000000', '--chunk', '200000',
        '--n-seeds', '5', '--blocks-per-chunk', '40', '--focused-samples', '200000',
        '--out', '.\results\raw\11_pbme_projected_P20.json')
    Invoke-PythonStage 'Projected-density PBME P30' `
        'results\raw\11_pbme_projected_P30.json' 'results\logs\11_P30.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '30', '--n-samples', '1200000', '--chunk', '200000',
        '--n-seeds', '6', '--blocks-per-chunk', '40', '--focused-samples', '200000',
        '--out', '.\results\raw\11_pbme_projected_P30.json')
    Invoke-PythonStage 'Projected-density PBME P70' `
        'results\raw\11_pbme_projected_P70.json' 'results\logs\11_P70.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '70', '--T', '857.5',
        '--n-samples', '1200000', '--chunk', '200000',
        '--n-seeds', '6', '--blocks-per-chunk', '40', '--focused-samples', '200000',
        '--out', '.\results\raw\11_pbme_projected_P70.json')
    Invoke-PythonStage 'Projected-density PBME P100' `
        'results\raw\11_pbme_projected_P100.json' 'results\logs\11_P100.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '100', '--n-samples', '1200000', '--chunk', '200000',
        '--n-seeds', '6', '--blocks-per-chunk', '40', '--focused-samples', '200000',
        '--out', '.\results\raw\11_pbme_projected_P100.json')

    Invoke-PythonStage 'Projected PBME coupling C=0.005' `
        'results\raw\11_pbme_projected_P40_C0.005.json' 'results\logs\11_P40_C0.005.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '40', '--coupling', '0.005', '--tag', '_C0.005',
        '--n-samples', '400000', '--chunk', '80000', '--n-seeds', '5',
        '--blocks-per-chunk', '16', '--focused-samples', '200000')
    Invoke-PythonStage 'Projected PBME coupling C=0.0075' `
        'results\raw\11_pbme_projected_P40_C0.0075.json' 'results\logs\11_P40_C0.0075.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '40', '--coupling', '0.0075', '--tag', '_C0.0075',
        '--n-samples', '400000', '--chunk', '80000', '--n-seeds', '5',
        '--blocks-per-chunk', '16', '--focused-samples', '200000')
    Invoke-PythonStage 'Projected PBME coupling C=0.03' `
        'results\raw\11_pbme_projected_P40_C0.03.json' 'results\logs\11_P40_C0.03.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '40', '--coupling', '0.03', '--tag', '_C0.03',
        '--n-samples', '400000', '--chunk', '80000', '--n-seeds', '5',
        '--blocks-per-chunk', '16', '--focused-samples', '200000')
    Invoke-PythonStage 'Projected PBME coupling C=0.06' `
        'results\raw\11_pbme_projected_P40_C0.06.json' 'results\logs\11_P40_C0.06.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '40', '--coupling', '0.06', '--tag', '_C0.06',
        '--n-samples', '400000', '--chunk', '80000', '--n-seeds', '5',
        '--blocks-per-chunk', '16', '--focused-samples', '200000')

    Invoke-PythonStage 'Independent phi-proposal PBME cross-check P40' `
        'results\raw\11_pbme_projected_P40_phi.json' 'results\logs\11_P40_phi.log' @(
        '.\scripts\11_pbme_projected.py', '--config', '.\configs\tully_default.yaml',
        '--P0', '40', '--scheme', 'wigner_seo', '--tag', '_phi',
        '--n-samples', '400000', '--chunk', '80000', '--n-seeds', '5',
        '--blocks-per-chunk', '16', '--skip-focused')

    Invoke-PythonStage 'PBME sample/seed statistics P20' `
        'results\raw\11b_pbme_seed_convergence_P20.json' 'results\logs\11b_P20.log' @(
        '.\scripts\11b_pbme_statistics.py', '--P0', '20')
    Invoke-PythonStage 'PBME sample/seed statistics P10' `
        'results\raw\11b_pbme_seed_convergence_P10.json' 'results\logs\11b_P10.log' @(
        '.\scripts\11b_pbme_statistics.py', '--P0', '10')
    Invoke-PythonStage 'PBME sample/seed statistics P30' `
        'results\raw\11b_pbme_seed_convergence_P30.json' 'results\logs\11b_P30.log' @(
        '.\scripts\11b_pbme_statistics.py', '--P0', '30')
    Invoke-PythonStage 'PBME sample/seed statistics P70' `
        'results\raw\11b_pbme_seed_convergence_P70.json' 'results\logs\11b_P70.log' @(
        '.\scripts\11b_pbme_statistics.py', '--P0', '70')
    Invoke-PythonStage 'PBME sample/seed statistics P100' `
        'results\raw\11b_pbme_seed_convergence_P100.json' 'results\logs\11b_P100.log' @(
        '.\scripts\11b_pbme_statistics.py', '--P0', '100')

    # The terminal PBME density resolved in phase space: the ensembles of the
    # projected-density stages above, trajectory for trajectory, binned on the
    # cells of the focused-PBME field of 10b (enlarged at P20 and P30 to hold
    # the parts of the QCLE and PBME densities that lie away from the packets),
    # then compared with the quantum and QCLE fields.  P10 is not compared: its
    # terminal fields carry structure finer than the stored grids.
    Invoke-PythonStage 'Projected PBME in phase space P20' `
        'results\raw\18_pbme_phase_space_P20.json' 'results\logs\18_P20.log' @(
        '.\scripts\18_pbme_phase_space.py', '--P0', '20', '--blocks-per-chunk', '4',
        '--extend-to', '-6', '32', '-20', '42')
    Invoke-PythonStage 'Projected PBME in phase space P30' `
        'results\raw\18_pbme_phase_space_P30.json' 'results\logs\18_P30.log' @(
        '.\scripts\18_pbme_phase_space.py', '--P0', '30', '--blocks-per-chunk', '10',
        '--extend-to', '5', '25', '12.5', '44')
    Invoke-PythonStage 'Projected PBME in phase space P70' `
        'results\raw\18_pbme_phase_space_P70.json' 'results\logs\18_P70.log' @(
        '.\scripts\18_pbme_phase_space.py', '--P0', '70')
    Invoke-PythonStage 'Projected PBME in phase space P100' `
        'results\raw\18_pbme_phase_space_P100.json' 'results\logs\18_P100.log' @(
        '.\scripts\18_pbme_phase_space.py', '--P0', '100')
    foreach ($regime in 20, 30, 70, 100) {
        Invoke-PythonStage "Phase-space comparison P$regime" `
            "results\raw\19_phase_space_comparison_P$regime.json" "results\logs\19_P$regime.log" @(
            '.\scripts\19_phase_space_comparison.py', '--P0', "$regime")
    }
    # Focused PBME: scripts\11 archives only the diabatic populations of its
    # focused branch.  The same ensemble (same seed, sampler, step and saving
    # interval) is propagated again to read out the adiabatic populations and
    # the coherence; each run checks that it reproduces those populations.
    foreach ($regime in 10, 20, 30, 40, 70, 100) {
        Invoke-PythonStage "Focused PBME observables P$regime" `
            "results\raw\20_focused_pbme_observables_P$regime.json" "results\logs\20_P$regime.log" @(
            '.\scripts\20_focused_pbme_observables.py', '--P0', "$regime")
    }

    Assert-AggregateInputs
    Invoke-PythonChecked 'Projected-response leakage reanalysis' @(
        '.\scripts\08_leakage_diagnostic.py', '--aggregate',
        '--out', '.\results\raw\12_leakage_reanalysis.json') `
        'results\logs\12_leakage_reanalysis.log'
    if (-not (Test-Path -LiteralPath (Join-Path $RawDir '12_coupling_scan_projected_pbme.json'))) {
        throw 'Leakage reanalysis did not create 12_coupling_scan_projected_pbme.json'
    }
    Assert-AllProductionInputs
}

function Invoke-Postprocess {
    # The manuscript reads its numbers from manuscript\paper_values.tex and its
    # figures from results\figures\paper_*.pdf, both written from results\raw
    # by the two scripts below; nothing in the text is typed by hand.  The
    # computer-algebra archive behind Table SI takes about a minute and is
    # rebuilt only when it is missing.
    if (-not (Test-Path -LiteralPath (Join-Path $RawDir '17_generator_structure.json'))) {
        Invoke-PythonChecked 'Computer-algebra checks of the generator' @(
            '.\scripts\17_generator_structure.py') 'results\logs\17_generator_structure.log'
    }
    Invoke-PythonChecked 'Paper values from the archived results' @(
        '.\scripts\15_paper_values.py') 'results\logs\15_paper_values.log'
    Invoke-PythonChecked 'Paper figures from the archived results' @(
        '.\scripts\16_paper_figures.py') 'results\logs\16_paper_figures.log'
    Invoke-PythonChecked 'Provenance table of the archived results' @(
        '.\scripts\21_provenance_table.py') 'results\logs\21_provenance_table.log'

    # The supplement reads the main text's equation numbers from its .aux file
    # (package xr), so the main text is compiled to convergence first.
    Push-Location -LiteralPath (Join-Path $ProjectRoot 'manuscript')
    try {
        Invoke-NativeChecked 'Main manuscript initial LaTeX pass' 'pdflatex' `
            @('-interaction=nonstopmode', '-halt-on-error', 'mqcle_hermite_closure.tex') `
            'results\logs\latex_main_initial.log'
        Invoke-NativeChecked 'Main manuscript bibliography' 'bibtex' `
            @('mqcle_hermite_closure') 'results\logs\bibtex_main.log'
        1..2 | ForEach-Object {
            Invoke-NativeChecked "Main manuscript pass $_" 'pdflatex' `
                @('-interaction=nonstopmode', '-halt-on-error', 'mqcle_hermite_closure.tex') `
                "results\logs\latex_main_pass$_.log"
        }
        Invoke-NativeChecked 'Supplement initial LaTeX pass' 'pdflatex' `
            @('-interaction=nonstopmode', '-halt-on-error', 'supplementary_material.tex') `
            'results\logs\latex_supp_initial.log'
        Invoke-NativeChecked 'Supplement bibliography' 'bibtex' `
            @('supplementary_material') 'results\logs\bibtex_supp.log'
        1..2 | ForEach-Object {
            Invoke-NativeChecked "Supplement pass $_" 'pdflatex' `
                @('-interaction=nonstopmode', '-halt-on-error', 'supplementary_material.tex') `
                "results\logs\latex_supp_pass$_.log"
        }
    }
    finally { Pop-Location }
}

function Write-Manifest {
    $lines = Get-ChildItem -LiteralPath $ProjectRoot -File -Recurse | Where-Object {
        $_.FullName -ne $Manifest -and
        $_.FullName -notmatch '\\(__pycache__|\.git|\.pytest_cache|_to_delete)\\'
    } | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($ProjectRoot.Length + 1).Replace('\', '/')
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }
    [IO.File]::WriteAllLines($Manifest, $lines)
    Write-Host "[done] SHA-256 manifest: $Manifest ($($lines.Count) files)"
}

function Invoke-Audit {
    Invoke-PythonChecked 'Full pytest suite' @('-m', 'pytest', '.\tests', '-q') `
        'results\logs\pytest_final.log'
    Invoke-PythonChecked 'JCP format audit' @('.\scripts\14_format_audit.py') `
        'results\logs\14_format_audit.log'
    $acceptanceArgs = @('.\scripts\13_acceptance.py')
    if ($Submission) { $acceptanceArgs += '--submission' }
    Invoke-PythonChecked 'Final acceptance audit' $acceptanceArgs `
        'results\logs\13_acceptance.log'
    Write-Manifest
}

function Show-Plan {
    @"
Every production stage has been run and its validated output is archived in
results\raw; each stage below is skipped when that output exists.  Outputs of
stages that this runner does not start (for example the computer-algebra
scripts and the four-field solver at P0 other than 40) record the command that
produced them in their metadata, and results\PROVENANCE.md lists every
archived output with its command.

Phases:
  Preflight    required files, Python dependencies, archived P40 PBME inputs
  Heavy        subsystem-QCLE reference (07, 07c), leakage diagnostics (08),
               full-quantum convergence (10), four-field cross-check at P40
               (10b), projected and focused PBME (11, 11b, 18, 19, 20) and the
               six-regime aggregate (08 --aggregate)
  Hermite      Gaussian-Hermite gates GH-01..GH-15, basis covariance (07cg),
               cross-check (10g), convergence (09g), closure validation (07k)
               and the coefficient propagation (07g) at P0 = 40, 30, 100, 70,
               20 and 10; then Postprocess and Audit
  Postprocess  computer algebra (17, if missing), paper values (15), paper
               figures (16), provenance table (21) and the LaTeX build of
               the manuscript
  Audit        pytest suite, JCP format audit (14), acceptance audit (13) and
               the SHA-256 manifest
  Postprocess and Audit need the manuscript sources in manuscript\.

07_coefficient_propagation.py advances the reduced field rho_W by Strang
splitting and is the SUBSYSTEM-QCLE REFERENCE.  The Gaussian-Hermite
coefficient system of the closure theorem is propagated by
07g_gh_coefficient_propagation.py on the threaded kernel of
src\mqcle\hermite\kernel.py, which gate GH-12 holds to the reference
implementation; -Threads N overrides os.cpu_count().  Outputs of the two
branches are never relabelled as one another.

Commands:
  & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$ProjectRoot\run_final_pipeline.ps1" -Phase Preflight
  & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$ProjectRoot\run_final_pipeline.ps1" -Phase Hermite
  & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$ProjectRoot\run_final_pipeline.ps1" -Phase Heavy
  & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$ProjectRoot\run_final_pipeline.ps1" -Phase Postprocess
  & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$ProjectRoot\run_final_pipeline.ps1" -Phase Audit

Or run every phase in sequence:
  & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$ProjectRoot\run_final_pipeline.ps1" -Phase All

Submission mode (only after authors replace every \needs{} marker):
  & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$ProjectRoot\run_final_pipeline.ps1" -Phase Audit -Submission
"@ | Write-Host
}

switch ($Phase) {
    'List'        { Show-Plan }
    'Preflight'   { Invoke-Preflight }
    'Hermite'     { Invoke-Hermite }
    'Heavy'       { Invoke-Heavy }
    'Postprocess' { Invoke-Preflight; Assert-AllProductionInputs; Invoke-Postprocess }
    'Audit'       { Invoke-Preflight; Assert-AllProductionInputs; Invoke-Audit }
    'All'         { Invoke-Heavy; Invoke-Hermite; Invoke-Postprocess; Invoke-Audit }
}
