# 定义要下载的年份
$years = 2017, 2018, 2019
# 定义最大重试次数
$maxRetries = 200

foreach ($year in $years) {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "正在处理 $year 年的数据..." -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    
    $i = 0
    while ($i -lt $maxRetries) {
        Write-Host "[$year] 正在进行第 $($i+1) 次同步尝试..."
        
        # 使用 sync 命令，目标路径如果不加年份子目录，所有文件会混在一起
        # 这里我没加子目录，是为了保持和你之前的路径结构一致 (./dataset/sevir_data/)
        # 如果你想按年份分文件夹，请把下一行最后改为 ./dataset/sevir_data/$year/
        aws s3 sync --no-sign-request s3://sevir/data/vil/$year/ ./dataset/sevir_data/
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ">>> $year 年数据下载成功！" -ForegroundColor Green
            break
        } else {
            Write-Host ">>> 连接中断，等待 5 秒后自动重试..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
            $i++
        }
    }
    
    if ($i -eq $maxRetries) {
        Write-Host "!!! $year 年数据下载多次失败，请检查网络。" -ForegroundColor Red
    }
}

Write-Host "所有任务处理完毕。" -ForegroundColor Magenta