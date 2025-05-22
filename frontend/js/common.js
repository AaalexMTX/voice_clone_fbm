// API基础URL
// 移除了"common.js - 包含公共函数和初始化逻辑
const API_BASE_URL = 'http://localhost:8083/api';

// 获取认证头部
function getAuthHeaders() {
    const token = localStorage.getItem('token');
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };
}

// 页面加载时检查登录状态并初始化
document.addEventListener('DOMContentLoaded', function () {
    checkAuth();
    loadUserInfo();

    // 根据当前激活的标签加载对应模块
    const activeTab = document.querySelector('.sidebar li.active');
    if (activeTab) {
        const tabId = activeTab.getAttribute('onclick').match(/switchTab\('(.+?)'\)/)[1];
        switchTab(tabId, false); // false表示不触发点击事件，只加载内容
    } else {
        loadAudioList(); // 默认加载音频列表
    }
});

// 检查登录状态
function checkAuth() {
    const user = localStorage.getItem('user');
    if (!user) {
        window.location.href = '/index.html';
    }
}

// 加载用户信息
function loadUserInfo() {
    const user = JSON.parse(localStorage.getItem('user'));
    if (user) {
        document.getElementById('username').textContent = user.nickname || user.username;
        document.getElementById('nickname').value = user.nickname || '';
        document.getElementById('email').value = user.email || '';
    }
}

// 切换标签页
function switchTab(tabId, isClick = true) {
    // 隐藏所有标签页
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // 显示选中的标签页
    document.getElementById(tabId).classList.add('active');

    // 更新侧边栏选中状态
    if (isClick) {
        document.querySelectorAll('.sidebar li').forEach(item => {
            item.classList.remove('active');
        });
        event.target.classList.add('active');
    }

    // 根据标签页加载对应内容
    switch (tabId) {
        case 'voice-clone':
            loadAudioList();
            break;
        case 'model-training':
            loadTrainingAudioList();
            break;
        case 'voice-synthesis':
            initSynthesisFeatures();
            loadUserModels();
            break;
        case 'settings':
            // 设置页面不需要额外加载
            break;
    }
}

// 退出登录
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/index.html';
}

// 显示消息提示
function showMessage(message, type = 'info') {
    const statusDiv = document.getElementById('status-message');
    if (!statusDiv) {
        const div = document.createElement('div');
        div.id = 'status-message';
        document.querySelector('.upload-area').appendChild(div);
    }

    const messageDiv = document.getElementById('status-message');
    messageDiv.textContent = message;
    messageDiv.className = `message ${type}`;

    // 3秒后自动消失
    setTimeout(() => {
        messageDiv.textContent = '';
        messageDiv.className = 'message';
    }, 3000);
}

// 格式化状态文本
function getStatusText(status) {
    const statusMap = {
        'pending': '待处理',
        'processing': '处理中',
        'completed': '已完成',
        'failed': '处理失败'
    };
    return statusMap[status] || status;
}

// 格式化日期
function formatDate(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// 格式化音频时长
function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
} 