param(
    [Parameter(Mandatory = $true)]
    [string]$ResultsPath,

    [string]$PatientLabel = "averageS",

    [double]$CorrXMin1 = 0.00,
    [double]$CorrXMax1 = 0.25,
    [double]$CorrXMin3 = 0.00,
    [double]$CorrXMax3 = 0.35,
    [double]$CorrXMin5 = 0.00,
    [double]$CorrXMax5 = 0.45,

    [double]$StripXMin = 0.10,
    [double]$StripXMax = 1.00,

    [double]$VarXMin1 = 0.0,
    [double]$VarXMax1 = 0.4,
    [double]$VarXMin3 = 0.0,
    [double]$VarXMax3 = 0.4,
    [double]$VarXMin5 = 0.0,
    [double]$VarXMax5 = 0.4,

    [double]$RbXMin = -60,
    [double]$RbXMax = 100
)

$ErrorActionPreference = "Stop"

$ArgsList = @(
    "run_full_pipeline.py",
    "--mode", "plots-only",
    "--results-path", $ResultsPath,
    "--patient-label", $PatientLabel,
    "--corr-xlim-1", $CorrXMin1, $CorrXMax1,
    "--corr-xlim-3", $CorrXMin3, $CorrXMax3,
    "--corr-xlim-5", $CorrXMin5, $CorrXMax5,
    "--strip-xlim", $StripXMin, $StripXMax,
    "--var-xlim-1", $VarXMin1, $VarXMax1,
    "--var-xlim-3", $VarXMin3, $VarXMax3,
    "--var-xlim-5", $VarXMin5, $VarXMax5,
    "--rb-xlim", $RbXMin, $RbXMax
)

python @ArgsList
