const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const pageIndicator = document.getElementById("pageIndicator");
const videoContainer = document.getElementById("videoContainer");
const videoPlayer = document.getElementById("videoPlayer");
const playPauseButton = document.getElementById("playPauseButton");
const rewindButton = document.getElementById("rewindButton");
const forwardButton = document.getElementById("forwardButton");
const backButton = document.getElementById("backButton");

// 设置 Canvas 大小为全屏
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// 常量
const COVER_MARGIN = 40; // 封面间距
const COVER_RADIUS = 15; // 封面圆角半径
const COLS = 4; // 每行封面数量
const ROWS = 3; // 每列封面数量
const COVERS_PER_PAGE = COLS * ROWS; // 每页封面数量
const DWELL_TIME = 1.5; // 停留点击时间（秒）
const PROGRESS_RADIUS = 40; // 进度条半径
const SCROLL_AREA_SIZE = 80; // 翻页区域大小
const SCROLL_AREA_MARGIN = 20; // 翻页区域边距
const VIDEO_FOLDER = "D:/users/annual_project_of_grade1/demo/demo/videos/"; // 视频文件夹路径
const COVER_DETECTION_PADDING = 20; // 封面检测区域大小
const PAGE_SCROLL_INTERVAL = 1000; // 翻页操作间隔时间（毫秒）

// 全局变量
let covers = []; // 所有封面数据
let currentPage = 0; // 当前页码
let currentCoverIndex = -1; // 当前选中的封面索引
let startTime = null; // 开始时间
let progress = 0; // 进度条进度
let isInVideoPage = false; // 是否在视频播放页
let currentVideo = null; // 当前播放的视频
let isVideoPlaying = true; // 视频是否正在播放
let lastScrollTime = 0; // 上次翻页时间
let hoveredButton = null; // 当前悬停的按钮
let buttonStartTime = null; // 按钮悬停开始时间
let buttonProgress = 0; // 按钮进度条进度
let coverImages = []; // 封面图片对象
let coverWidth = 0; // 动态封面宽度
let coverHeight = 0; // 动态封面高度

// 计算动态封面尺寸
function calculateCoverDimensions() {
    // 计算可用空间（考虑边距）
    const availableWidth = canvas.width - (COLS + 1) * COVER_MARGIN;
    const availableHeight = canvas.height - (ROWS + 1) * COVER_MARGIN;
    
    // 计算单个封面的最大尺寸
    const maxWidth = availableWidth / COLS;
    const maxHeight = availableHeight / ROWS;
    
    // 使用较小的值作为基准，保持16:9的宽高比
    const aspectRatio = 16 / 9;
    if (maxWidth / aspectRatio <= maxHeight) {
        coverWidth = maxWidth;
        coverHeight = maxWidth / aspectRatio;
    } else {
        coverHeight = maxHeight;
        coverWidth = maxHeight * aspectRatio;
    }
}

// 监听窗口大小变化
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    calculateCoverDimensions();
});

// 初始化封面数据
function initCovers() {
    calculateCoverDimensions(); // 初始化时计算封面尺寸
    for (let i = 1; i <= 20; i++) {
        const cover = {
            id: i,
            title: `视频 ${i}`,
            thumbnail: `cover${i}.jpg`,
            video: `video${i}.mp4`
        };
        covers.push(cover);

        // 加载封面图片
        const img = new Image();
        img.src = VIDEO_FOLDER + cover.thumbnail;
        img.onload = () => {
            coverImages[i - 1] = img;
        };
        img.onerror = (e) => {
            console.error(`Failed to load image: ${cover.thumbnail}`, e);
        };
    }
}

// 绘制圆角矩形
function drawRoundedRect(x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
    ctx.fill();
}

// 绘制封面
function drawCovers() {
    const startIndex = currentPage * COVERS_PER_PAGE;
    const endIndex = Math.min(startIndex + COVERS_PER_PAGE, covers.length);

    // 计算封面区域的起始位置（居中）
    const totalWidth = COLS * coverWidth + (COLS - 1) * COVER_MARGIN;
    const totalHeight = ROWS * coverHeight + (ROWS - 1) * COVER_MARGIN;
    const startX = (canvas.width - totalWidth) / 2;
    const startY = (canvas.height - totalHeight) / 2;

    for (let i = startIndex; i < endIndex; i++) {
        const cover = covers[i];
        const row = Math.floor((i - startIndex) / COLS);
        const col = (i - startIndex) % COLS;
        const x = startX + col * (coverWidth + COVER_MARGIN);
        const y = startY + row * (coverHeight + COVER_MARGIN);

        // 绘制封面背景
        ctx.fillStyle = "#fff";
        drawRoundedRect(x, y, coverWidth, coverHeight, COVER_RADIUS);
        ctx.shadowColor = "rgba(0, 0, 0, 0.1)";
        ctx.shadowBlur = 10;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 4;

        // 绘制封面图片
        if (coverImages[i]) {
            ctx.save();
            ctx.beginPath();
            ctx.roundRect(x, y, coverWidth, coverHeight, COVER_RADIUS);
            ctx.clip();
            
            // 使用高质量图像缩放
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            
            // 计算图片绘制参数以保持原始比例
            const img = coverImages[i];
            const imgRatio = img.width / img.height;
            const coverRatio = coverWidth / coverHeight;
            
            let drawWidth, drawHeight, drawX, drawY;
            
            if (imgRatio > coverRatio) {
                // 图片更宽，以高度为基准
                drawHeight = coverHeight;
                drawWidth = drawHeight * imgRatio;
                drawX = x + (coverWidth - drawWidth) / 2;
                drawY = y;
            } else {
                // 图片更高，以宽度为基准
                drawWidth = coverWidth;
                drawHeight = drawWidth / imgRatio;
                drawX = x;
                drawY = y + (coverHeight - drawHeight) / 2;
            }

            // 创建临时画布进行高质量缩放
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d', { alpha: false });
            
            // 设置临时画布尺寸为实际显示尺寸的2倍，用于提高清晰度
            const scale = 2;
            tempCanvas.width = drawWidth * scale;
            tempCanvas.height = drawHeight * scale;
            
            // 在临时画布上使用高质量设置绘制图片
            tempCtx.imageSmoothingEnabled = true;
            tempCtx.imageSmoothingQuality = 'high';
            tempCtx.drawImage(img, 0, 0, tempCanvas.width, tempCanvas.height);
            
            // 将临时画布内容绘制到主画布
            ctx.drawImage(tempCanvas, drawX, drawY, drawWidth, drawHeight);
            
            ctx.restore();
        }

        // 绘制封面标题
        ctx.fillStyle = "#333";
        ctx.font = "18px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(cover.title, x + coverWidth / 2, y + coverHeight + 30);

        // 绘制进度条
        if (i === currentCoverIndex) {
            const startAngle = -Math.PI / 2;
            const endAngle = startAngle + Math.PI * 2 * progress;
            ctx.beginPath();
            ctx.arc(x + coverWidth / 2, y + coverHeight / 2, PROGRESS_RADIUS, startAngle, endAngle);
            ctx.strokeStyle = "#007AFF";
            ctx.lineWidth = 5;
            ctx.stroke();
            ctx.closePath();
        }
    }
}

// 绘制翻页三角区域
function drawScrollAreas() {
    // 右上角（向上翻页）
    ctx.beginPath();
    ctx.moveTo(canvas.width, 0);
    ctx.lineTo(canvas.width - SCROLL_AREA_SIZE, 0);
    ctx.lineTo(canvas.width, SCROLL_AREA_SIZE);
    ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
    ctx.fill();
    ctx.closePath();

    // 右下角（向下翻页）
    ctx.beginPath();
    ctx.moveTo(canvas.width, canvas.height);
    ctx.lineTo(canvas.width - SCROLL_AREA_SIZE, canvas.height);
    ctx.lineTo(canvas.width, canvas.height - SCROLL_AREA_SIZE);
    ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
    ctx.fill();
    ctx.closePath();
}

// 进入视频播放页
function enterVideoPage(videoSrc) {
    isInVideoPage = true;
    videoContainer.style.display = "flex";
    videoPlayer.src = VIDEO_FOLDER + videoSrc;
    
    // 重置所有按钮的悬停状态
    hoveredButton = null;
    buttonStartTime = null;
    buttonProgress = 0;
    const buttons = [backButton, playPauseButton, rewindButton, forwardButton];
    buttons.forEach(button => {
        updateButtonProgress(button, 0);
    });
    
    // 设置视频播放质量
    videoPlayer.setAttribute('playsinline', '');
    videoPlayer.setAttribute('webkit-playsinline', '');
    videoPlayer.setAttribute('x5-playsinline', '');
    videoPlayer.setAttribute('x5-video-player-type', 'h5');
    videoPlayer.setAttribute('x5-video-player-fullscreen', 'true');
    videoPlayer.setAttribute('x5-video-orientation', 'portraint');
    videoPlayer.setAttribute('preload', 'auto');
    
    // 添加多个事件监听器来确保视频播放
    const playVideo = () => {
        const playPromise = videoPlayer.play();
        if (playPromise !== undefined) {
            playPromise.then(() => {
                // 播放成功
                updatePlayPauseButton();
            }).catch(error => {
                // 播放失败，尝试静音播放
                videoPlayer.muted = true;
                videoPlayer.play().then(() => {
                    updatePlayPauseButton();
                });
            });
        }
    };

    // 在视频加载完成后尝试播放
    videoPlayer.onloadeddata = playVideo;
    // 在视频可以播放时尝试播放
    videoPlayer.oncanplay = playVideo;
    // 立即尝试播放
    playVideo();
}

// 退出视频播放页
function exitVideoPage() {
    isInVideoPage = false;
    videoContainer.style.display = "none";
    videoPlayer.pause();
    videoPlayer.src = "";
}

// 更新播放/暂停按钮图标
function updatePlayPauseButton() {
    const icon = videoPlayer.paused ? "icons/play.png" : "icons/pause.png";
    playPauseButton.querySelector("img").src = icon;
}

// 检查是否在翻页区域
function isInScrollArea(mouseX, mouseY) {
    // 检查是否在右侧翻页区域内
    if (mouseX < canvas.width - SCROLL_AREA_MARGIN) {
        return false;
    }

    // 检查是否在右上角翻页区域
    if (mouseY < SCROLL_AREA_SIZE) {
        return {
            area: 'top',
            inArea: true
        };
    }

    // 检查是否在右下角翻页区域
    if (mouseY > canvas.height - SCROLL_AREA_SIZE) {
        return {
            area: 'bottom',
            inArea: true
        };
    }

    return false;
}

// 检查鼠标是否在封面附近
function checkCoverProximity(mouseX, mouseY) {
    if (isInVideoPage) return -1; // 在播放页时不检测封面

    // 如果鼠标在翻页区域内，不检测封面
    if (isInScrollArea(mouseX, mouseY)) {
        return -1;
    }

    const startIndex = currentPage * COVERS_PER_PAGE;
    const endIndex = Math.min(startIndex + COVERS_PER_PAGE, covers.length);

    // 计算封面区域的起始位置（居中）
    const totalWidth = COLS * coverWidth + (COLS - 1) * COVER_MARGIN;
    const totalHeight = ROWS * coverHeight + (ROWS - 1) * COVER_MARGIN;
    const startX = (canvas.width - totalWidth) / 2;
    const startY = (canvas.height - totalHeight) / 2;

    for (let i = startIndex; i < endIndex; i++) {
        const cover = covers[i];
        const row = Math.floor((i - startIndex) / COLS);
        const col = (i - startIndex) % COLS;
        const x = startX + col * (coverWidth + COVER_MARGIN);
        const y = startY + row * (coverHeight + COVER_MARGIN);

        if (
            mouseX >= x - COVER_DETECTION_PADDING &&
            mouseX <= x + coverWidth + COVER_DETECTION_PADDING &&
            mouseY >= y - COVER_DETECTION_PADDING &&
            mouseY <= y + coverHeight + COVER_DETECTION_PADDING
        ) {
            return i;
        }
    }
    return -1;
}

// 更新按钮进度条
function updateButtonProgress(button, progress) {
    const circle = button.querySelector("circle");
    const circumference = 2 * Math.PI * 70;
    const dashLength = (progress * circumference).toFixed(2);
    circle.style.strokeDasharray = `${dashLength} ${circumference}`;
}

// 处理按钮悬停逻辑
function handleButtonHover(button, action) {
    const rect = button.getBoundingClientRect();
    const padding = 70; // 增加检测区域大小
    const isHovered =
        window.mouseX >= rect.left - padding &&
        window.mouseX <= rect.right + padding &&
        window.mouseY >= rect.top - padding &&
        window.mouseY <= rect.bottom + padding;

    if (isHovered) {
        if (hoveredButton === button) {
            const elapsedTime = (Date.now() - buttonStartTime) / 1000;
            buttonProgress = elapsedTime / DWELL_TIME;
            updateButtonProgress(button, buttonProgress);
            if (elapsedTime >= DWELL_TIME) {
                action();
                hoveredButton = null;
                buttonStartTime = null;
                buttonProgress = 0;
                updateButtonProgress(button, 0);
            }
        } else {
            hoveredButton = button;
            buttonStartTime = Date.now();
            buttonProgress = 0;
            updateButtonProgress(button, 0);
        }
    } else if (hoveredButton === button) {
        hoveredButton = null;
        buttonStartTime = null;
        buttonProgress = 0;
        updateButtonProgress(button, 0);
    }
}

// 主循环
function gameLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (isInVideoPage) {
        // 视频播放页不需要绘制 Canvas 内容
    } else {
        drawCovers();
        drawScrollAreas();
    }

    // 获取鼠标位置
    const mouseX = window.mouseX || 0;
    const mouseY = window.mouseY || 0;

    if (!isInVideoPage) {
        // 首先检查是否在翻页区域
        const scrollArea = isInScrollArea(mouseX, mouseY);
        if (scrollArea) {
            if (scrollArea.area === 'top') {
                // 向上翻页
                currentPage = Math.max(currentPage - 1, 0);
            } else {
                // 向下翻页
                currentPage = Math.min(currentPage + 1, Math.floor((covers.length - 1) / COVERS_PER_PAGE));
            }
            pageIndicator.textContent = `Page ${currentPage + 1}`;
            // 重置封面选择状态
            currentCoverIndex = -1;
            progress = 0;
        } else {
            // 不在翻页区域时，检查封面选择
            const coverIndex = checkCoverProximity(mouseX, mouseY);
            if (coverIndex !== -1) {
                if (currentCoverIndex !== coverIndex) {
                    currentCoverIndex = coverIndex;
                    startTime = Date.now();
                    progress = 0;
                } else {
                    const elapsedTime = (Date.now() - startTime) / 1000;
                    progress = elapsedTime / DWELL_TIME;
                    if (elapsedTime >= DWELL_TIME) {
                        enterVideoPage(covers[coverIndex].video);
                        currentCoverIndex = -1;
                        progress = 0;
                    }
                }
            } else {
                currentCoverIndex = -1;
                progress = 0;
            }
        }
    }

    // 检查鼠标是否在播放页按钮上
    if (isInVideoPage) {
        const buttons = [backButton, playPauseButton, rewindButton, forwardButton];
        buttons.forEach(button => {
            const rect = button.getBoundingClientRect();
            const buttonPadding = 70; // 按钮检测区域大小
            if (mouseX >= rect.left - buttonPadding && 
                mouseX <= rect.right + buttonPadding && 
                mouseY >= rect.top - buttonPadding && 
                mouseY <= rect.bottom + buttonPadding) {
                if (button === playPauseButton) {
                    handleButtonHover(button, () => {
                        if (videoPlayer.paused) {
                            videoPlayer.play();
                        } else {
                            videoPlayer.pause();
                        }
                        updatePlayPauseButton();
                    });
                } else if (button === rewindButton) {
                    handleButtonHover(button, () => {
                        videoPlayer.currentTime -= 10;
                    });
                } else if (button === forwardButton) {
                    handleButtonHover(button, () => {
                        videoPlayer.currentTime += 10;
                    });
                } else if (button === backButton) {
                    handleButtonHover(button, () => {
                        exitVideoPage();
                    });
                }
            } else if (hoveredButton === button) {
                hoveredButton = null;
                buttonStartTime = null;
                buttonProgress = 0;
                updateButtonProgress(button, 0);
            }
        });
    }

    requestAnimationFrame(gameLoop);
}

// 监听鼠标移动事件
window.addEventListener("mousemove", (event) => {
    window.mouseX = event.clientX;
    window.mouseY = event.clientY;
});

// 监听视频播放状态变化
videoPlayer.addEventListener("play", () => {
    updatePlayPauseButton();
});

videoPlayer.addEventListener("pause", () => {
    updatePlayPauseButton();
});

// 初始化封面数据
initCovers();

// 启动游戏循环
gameLoop();