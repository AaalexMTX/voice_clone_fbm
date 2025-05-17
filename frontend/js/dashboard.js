// API基础URL
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
    loadAudioList();
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
function switchTab(tabId) {
    // 隐藏所有标签页
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // 显示选中的标签页
    document.getElementById(tabId).classList.add('active');

    // 更新侧边栏选中状态
    document.querySelectorAll('.sidebar li').forEach(item => {
        item.classList.remove('active');
    });
    event.target.classList.add('active');
}

// 退出登录
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/index.html';
}

// 保存设置
async function saveSettings() {
    const nickname = document.getElementById('nickname').value;
    const email = document.getElementById('email').value;

    try {
        const response = await fetch(`${API_BASE_URL}/user/info`, {
            method: 'PUT',
            headers: getAuthHeaders(),
            body: JSON.stringify({ nickname, email }),
        });

        const data = await response.json();
        if (response.ok) {
            alert('设置保存成功');
            // 更新本地存储的用户信息
            const user = JSON.parse(localStorage.getItem('user'));
            const updatedUser = { ...user, nickname, email };
            localStorage.setItem('user', JSON.stringify(updatedUser));
            loadUserInfo();
        } else {
            alert(data.error || '保存失败');
        }
    } catch (error) {
        alert('网络错误，请稍后重试');
    }
}

// 加载音频列表
async function loadAudioList() {
    try {
        const response = await fetch(`${API_BASE_URL}/audio/list`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        const data = await response.json();
        if (response.ok) {
            displayAudioList(data.audios);
        } else {
            alert(data.error || '获取音频列表失败');
        }
    } catch (error) {
        alert('网络错误，请稍后重试');
    }
}

// 显示音频列表
function displayAudioList(audios) {
    const container = document.querySelector('.voice-list');
    container.innerHTML = '';

    if (!audios.length) {
        container.innerHTML = '<div class="no-audio">暂无音频文件</div>';
        return;
    }

    audios.forEach(audio => {
        const audioElement = document.createElement('div');
        audioElement.className = 'voice-item';
        audioElement.innerHTML = `
            <div class="voice-info">
                <span class="voice-name">${audio.name}</span>
                <span class="voice-status ${audio.status}">${getStatusText(audio.status)}</span>
                <span class="voice-date">${formatDate(audio.createdAt)}</span>
            </div>
            <div class="voice-controls">
                <audio src="${API_BASE_URL}/audio/stream/${audio.id}" controls preload="none"></audio>
                <button class="delete-btn" onclick="deleteAudio(${audio.id})">
                    <i class="fas fa-trash"></i> 删除
                </button>
            </div>
            <div class="voice-content">
                <textarea
                    class="content-textarea"
                    placeholder="请输入音频对应的文本内容"
                    onchange="updateAudioContent(${audio.id}, this.value)"
                >${audio.content || ''}</textarea>
            </div>
        `;
        container.appendChild(audioElement);
    });
}

// 删除音频
async function deleteAudio(id) {
    if (!confirm('确定要删除这个音频文件吗？')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/audio/${id}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            alert('删除成功');
            loadAudioList();
        } else {
            const data = await response.json();
            alert(data.error || '删除失败');
        }
    } catch (error) {
        alert('网络错误，请稍后重试');
    }
}

// 开始克隆
async function startClone() {
    const fileInput = document.getElementById('voice-file');
    const uploadStatus = document.getElementById('upload-status');

    if (!fileInput.files.length) {
        showMessage('请先选择音频文件', 'error');
        return;
    }

    const file = fileInput.files[0];
    // 检查文件类型
    if (!file.type.startsWith('audio/')) {
        showMessage('请上传音频文件（MP3或WAV格式）', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('audio', file);

    try {
        // 显示上传状态
        showMessage('正在上传音频文件...', 'info');

        const response = await fetch(`${API_BASE_URL}/audio/upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: formData,
        });

        const data = await response.json();
        if (response.ok) {
            showMessage('音频上传成功！', 'success');
            console.log('音频上传成功:', data);  // 添加日志

            // 清空文件输入
            fileInput.value = '';

            // 重新加载音频列表
            await loadAudioList();

            // 滚动到音频列表
            document.querySelector('.voice-list').scrollIntoView({ behavior: 'smooth' });
        } else {
            showMessage(data.error || '上传失败', 'error');
            console.error('上传失败:', data.error);  // 添加错误日志
        }
    } catch (error) {
        showMessage('网络错误，请稍后重试', 'error');
        console.error('上传错误:', error);  // 添加错误日志
    }
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

// 文件上传区域的拖放功能
const uploadArea = document.querySelector('.upload-area');
const fileInput = document.getElementById('voice-file');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = '#f1f2f6';
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'none';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'none';

    const files = e.dataTransfer.files;
    if (files.length) {
        fileInput.files = files;
    }
});

// 更新音频内容
async function updateAudioContent(audioId, content) {
    try {
        const response = await fetch(`${API_BASE_URL}/audio/${audioId}/content`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({ content })
        });

        const data = await response.json();
        if (response.ok) {
            showMessage('内容更新成功', 'success');
            console.log('音频内容更新成功:', data);
        } else {
            showMessage(data.error || '更新失败', 'error');
            console.error('更新失败:', data.error);
        }
    } catch (error) {
        showMessage('网络错误，请稍后重试', 'error');
        console.error('更新错误:', error);
    }
} 