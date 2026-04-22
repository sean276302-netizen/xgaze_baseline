const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

// 设置 Canvas 大小为全屏
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// 常量
const TARGET_RADIUS = 50;
const HIGHLIGHT_RADIUS = 50;
const DWELL_TIME = 1; // 持续停留时间（秒）
const PROGRESS_RADIUS = 60; // 扇形进度条的半径
const FPS = 60; // 帧率

// 全局变量
let currentTarget = createTarget();
let startTime = null;
let highlighted = false;
let progress = 0;

// 创建目标
function createTarget() {
    const x = Math.random() * (canvas.width - TARGET_RADIUS * 2) + TARGET_RADIUS;
    const y = Math.random() * (canvas.height - TARGET_RADIUS * 2) + TARGET_RADIUS;
    return { x, y };
}

// 绘制目标
function drawTarget(target) {
    ctx.beginPath();
    ctx.arc(target.x, target.y, TARGET_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = highlighted ? "yellow" : "red";
    ctx.fill();
    ctx.closePath();
}

// 绘制进度条
function drawProgress(target, progress) {
    if (progress < 1) {
        const startAngle = -Math.PI / 2; // 从12点钟方向开始
        const endAngle = startAngle + Math.PI * 2 * progress; // 顺时针绘制
        ctx.beginPath();
        ctx.arc(target.x, target.y, PROGRESS_RADIUS, startAngle, endAngle);
        ctx.strokeStyle = "blue";
        ctx.lineWidth = 5;
        ctx.stroke();
        ctx.closePath();
    }
}

// 检查鼠标是否在目标附近
function checkProximity(mouseX, mouseY, target) {
    const distance = Math.hypot(mouseX - target.x, mouseY - target.y);
    if (distance <= HIGHLIGHT_RADIUS) {
        if (!highlighted) {
            highlighted = true;
            startTime = Date.now();
        } else {
            const elapsedTime = (Date.now() - startTime) / 1000; // 转换为秒
            progress = elapsedTime / DWELL_TIME;
            if (elapsedTime >= DWELL_TIME) {
                return true; // 目标被消灭
            }
        }
    } else {
        if (highlighted) {
            highlighted = false;
            progress = 0;
        }
    }
    return false;
}

// 清屏
function clearScreen() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// 主循环
function gameLoop() {
    clearScreen();

    // 获取鼠标位置
    const mouseX = window.mouseX || 0;
    const mouseY = window.mouseY || 0;

    // 检查鼠标是否在目标附近
    if (checkProximity(mouseX, mouseY, currentTarget)) {
        currentTarget = createTarget();
        highlighted = false;
        progress = 0;
    }

    // 绘制目标
    drawTarget(currentTarget);

    // 绘制进度条
    if (highlighted) {
        drawProgress(currentTarget, progress);
    }

    // 循环调用
    requestAnimationFrame(gameLoop);
}

// 监听鼠标移动事件
window.addEventListener("mousemove", (event) => {
    window.mouseX = event.clientX;
    window.mouseY = event.clientY;
});

// 监听键盘事件
window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" || event.key === "q") {
        alert("退出测试");
        window.close(); // 关闭窗口（部分浏览器可能不支持）
    }
});

// 启动游戏循环
gameLoop();