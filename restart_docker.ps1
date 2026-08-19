param (
    [switch]$Detach = $false
)

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   Pipecat Docker Service Manager" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Function to check if Docker daemon is responsive
function Test-DockerDaemon {
    $null = docker info 2>&1
    return ($LASTEXITCODE -eq 0)
}

# 2. Check & Launch Docker Desktop if needed
if (-not (Test-DockerDaemon)) {
    Write-Host "[1/3] Docker daemon is not running." -ForegroundColor Yellow

    $dockerProcess = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
    if (-not $dockerProcess) {
        $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerPath) {
            Write-Host "      Launching Docker Desktop from '$dockerPath'..." -ForegroundColor Cyan
            Start-Process -FilePath $dockerPath
        } else {
            Write-Host "      Docker Desktop executable not found at '$dockerPath'." -ForegroundColor Red
            Write-Host "      Please start Docker Desktop manually." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "      Docker Desktop is starting up..." -ForegroundColor Cyan
    }

    Write-Host "      Waiting for Docker engine to be ready..." -NoNewline -ForegroundColor Gray
    $timeoutSeconds = 90
    $startTime = Get-Date

    while (-not (Test-DockerDaemon)) {
        if ((Get-Date) - $startTime -gt (New-TimeSpan -Seconds $timeoutSeconds)) {
            Write-Host ""
            Write-Host "      Timed out waiting for Docker engine after $timeoutSeconds seconds." -ForegroundColor Red
            Write-Host "      Please check Docker Desktop status and try again." -ForegroundColor Red
            exit 1
        }
        Write-Host "." -NoNewline -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
    Write-Host " Ready!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[1/3] Docker daemon is active and responsive." -ForegroundColor Green
    Write-Host ""
}

# 3. Stop existing containers
Write-Host "[2/3] Stopping existing containers (docker compose down)..." -ForegroundColor Cyan
docker compose down
if ($LASTEXITCODE -ne 0) {
    Write-Host "      docker compose down completed with non-zero status, proceeding..." -ForegroundColor Yellow
}
Write-Host ""

# 4. Build and start containers
Write-Host "[3/3] Building and starting containers (docker compose up --build)..." -ForegroundColor Cyan
Write-Host "      Web UI will be available at: http://localhost:7860" -ForegroundColor Green
Write-Host ""

if ($Detach) {
    docker compose up -d --build
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Containers started in detached mode. Access at http://localhost:7860" -ForegroundColor Green
    }
} else {
    docker compose up --build
}
