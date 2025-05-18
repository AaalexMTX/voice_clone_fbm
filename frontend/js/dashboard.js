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
    loadTrainingAudioList(); // 添加这行，确保页面加载时就加载训练音频列表
    initSynthesisFeatures();
    loadUserModels();
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

    // 如果切换到语音克隆标签页，刷新音频列表
    if (tabId === 'voice-clone') {
        loadAudioList();
    }

    // 如果切换到模型训练标签页，加载训练音频列表
    if (tabId === 'model-training') {
        loadTrainingAudioList();
    }
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
        container.innerHTML = '<div class="no-audio">🎵 暂无音频文件，请先上传音频</div>';
        return;
    }

    audios.forEach((audio, index) => {
        const audioElement = document.createElement('div');
        audioElement.className = 'voice-item';
        audioElement.innerHTML = `
            <div class="voice-player">
                <div class="audio-name">🎧 ${audio.name || `音频${index + 1}`}</div>
                <audio src="${API_BASE_URL}/audio/stream/${audio.id}" controls preload="none"></audio>
            </div>
            <div class="voice-content">
                <textarea
                    class="content-textarea"
                    placeholder="请输入需要校对的文本..."
                >${audio.content || ''}</textarea>
            </div>
            <div class="voice-actions">
                <button class="update-btn" onclick="updateAudioContent(${audio.id}, this.parentElement.previousElementSibling.querySelector('.content-textarea').value)">
                    ✓ 更新文本
                </button>
                <button class="delete-btn" onclick="deleteAudio(${audio.id})">
                    ✕ 删除音频
                </button>
            </div>
        `;
        container.appendChild(audioElement);

        // 添加淡入动画延迟
        audioElement.style.animationDelay = `${index * 0.1}s`;
    });
}

// 生成波形动画HTML
function generateWaveAnimation() {
    return Array(20).fill().map(() =>
        `<div class="wave-bar"></div>`
    ).join('');
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
        uploadAudio(files[0]); // 直接上传文件，不等待点击"开始克隆"
    }
});

// 文件选择变化时自动上传
fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length > 0) {
        uploadAudio(fileInput.files[0]);
    }
});

// 上传音频文件
async function uploadAudio(file) {
    try {
        // 检查文件类型
        if (!file.type.startsWith('audio/')) {
            showMessage('请上传音频文件（MP3或WAV格式）', 'error');
            return;
        }

        // 显示文件预览
        showFilePreview(file);

        showMessage('正在上传音频文件...', 'info');

        const formData = new FormData();
        formData.append('audio', file);

        // 创建 XMLHttpRequest 对象以支持进度显示
        const xhr = new XMLHttpRequest();
        const promise = new Promise((resolve, reject) => {
            xhr.upload.onprogress = function (e) {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    updateUploadProgress(percentComplete);
                }
            };

            xhr.onload = function () {
                if (xhr.status === 200) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(xhr.responseText));
                }
            };

            xhr.onerror = function () {
                reject(new Error('网络错误'));
            };
        });

        xhr.open('POST', `${API_BASE_URL}/audio/upload`);
        xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('token')}`);
        xhr.send(formData);

        const data = await promise;
        showMessage('音频上传成功！', 'success');
        console.log('音频上传成功:', data);

        // 清空文件输入和进度条
        fileInput.value = '';
        clearUploadProgress();

        // 重新加载音频列表
        await loadAudioList();

        // 滚动到音频列表
        document.querySelector('.voice-list').scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        showMessage('网络错误，请稍后重试', 'error');
        console.error('上传错误:', error);
        clearUploadProgress();
    }
}

// 显示文件预览
function showFilePreview(file) {
    const uploadArea = document.querySelector('.upload-area');
    uploadArea.innerHTML = `
        <div class="file-preview">
            <div class="file-info">
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
            <div class="progress-bar">
                <div class="progress" style="width: 0%"></div>
            </div>
            <button class="cancel-upload" onclick="cancelUpload()">取消</button>
        </div>
    `;
}

// 更新上传进度
function updateUploadProgress(percent) {
    const progress = document.querySelector('.progress');
    if (progress) {
        progress.style.width = `${percent}%`;
    }
}

// 清除上传进度和预览
function clearUploadProgress() {
    const uploadArea = document.querySelector('.upload-area');
    uploadArea.innerHTML = `
        <span>点击或拖拽上传训练音频</span>
        <small>支持 WAV, MP3 格式，建议上传5-10分钟的清晰语音</small>
        <div id="status-message" class="message"></div>
    `;
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 取消上传
function cancelUpload() {
    // 清除文件输入
    fileInput.value = '';
    clearUploadProgress();
    showMessage('已取消上传', 'info');
}

// 开始训练
async function startClone() {
    const fileInput = document.getElementById('voice-file');

    // 检查是否有选择文件
    if (!fileInput.files.length) {
        showMessage('请先选择或上传训练音频', 'error');
        return;
    }

    // 这里添加开始训练的业务逻辑
    showMessage('正在开始模型训练，请耐心等待...', 'info');

    // 后续可以添加调用训练API的逻辑
    // const response = await fetch(`${API_BASE_URL}/audio/train`, ...);
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

// 更新音频内容
async function updateAudioContent(audioId, content) {
    try {
        showMessage('正在保存文本...', 'info');

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
            showMessage('文本保存成功', 'success');
            console.log('音频内容更新成功:', data);
        } else {
            showMessage(data.error || '保存失败', 'error');
            console.error('更新失败:', data.error);
        }
    } catch (error) {
        showMessage('网络错误，请稍后重试', 'error');
        console.error('更新错误:', error);
    }
}

// 加载训练音频列表
async function loadTrainingAudioList() {
    try {
        const response = await fetch(`${API_BASE_URL}/audio/list`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        const data = await response.json();
        if (response.ok) {
            displayTrainingAudioList(data.audios);
        } else {
            alert(data.error || '获取音频列表失败');
        }
    } catch (error) {
        alert('网络错误，请稍后重试');
    }
}

// 显示训练音频列表
function displayTrainingAudioList(audios) {
    const container = document.getElementById('training-audio-list');
    if (!container) return; // 添加检查，防止在其他页面报错

    container.innerHTML = '';

    if (!audios.length) {
        container.innerHTML = '<div class="no-audio">暂无可用的训练音频，请先上传音频文件</div>';
        return;
    }

    audios.forEach(audio => {
        const audioElement = document.createElement('div');
        audioElement.className = 'audio-item';
        audioElement.innerHTML = `
            <div class="audio-checkbox">
                <input type="checkbox" id="audio-${audio.id}" value="${audio.id}">
                <div class="checkbox-custom"></div>
            </div>
            <div class="audio-info">
                <label class="audio-name" for="audio-${audio.id}">${audio.name}</label>
                <div class="audio-details">
                    <span class="audio-duration">${formatDuration(audio.duration || 0)}</span>
                    <span class="audio-status ${audio.status}">${getStatusText(audio.status)}</span>
                </div>
            </div>
        `;
        container.appendChild(audioElement);
    });
}

// 格式化音频时长
function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// 显示训练状态
function showTrainingStatus(message, type = 'info') {
    const statusDiv = document.getElementById('training-status');
    statusDiv.textContent = '';

    // 如果是加载状态，添加加载动画
    if (type === 'loading') {
        const loadingSpinner = document.createElement('div');
        loadingSpinner.className = 'loading';
        statusDiv.appendChild(loadingSpinner);
    }

    const messageSpan = document.createElement('span');
    messageSpan.textContent = message;
    statusDiv.appendChild(messageSpan);

    statusDiv.className = `training-status ${type}`;

    // 移除之前的show类
    statusDiv.classList.remove('show');

    // 强制重绘
    void statusDiv.offsetWidth;

    // 添加show类触发动画
    statusDiv.classList.add('show');
}

// 开始训练模型
async function startTraining() {
    // 获取选中的音频
    const selectedAudios = Array.from(document.querySelectorAll('#training-audio-list input[type="checkbox"]:checked'))
        .map(checkbox => checkbox.value);

    if (selectedAudios.length === 0) {
        showTrainingStatus('请选择至少一个训练音频', 'error');
        return;
    }

    // 获取模型名称
    const modelName = document.getElementById('model-name').value.trim();
    if (!modelName) {
        showTrainingStatus('请输入模型名称', 'error');
        return;
    }

    // 获取训练参数
    const trainingParams = {
        epochs: parseInt(document.getElementById('epochs').value),
        batchSize: parseInt(document.getElementById('batch-size').value),
        learningRate: parseFloat(document.getElementById('learning-rate').value)
    };

    // 显示训练开始状态
    showTrainingStatus('正在准备开始训练...', 'loading');

    try {
        const response = await fetch(`${API_BASE_URL}/model/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                modelName,
                audioIds: selectedAudios,
                params: trainingParams
            })
        });

        const data = await response.json();
        if (response.ok) {
            showTrainingStatus('模型训练已开始，请耐心等待...', 'success');
            pollTrainingStatus(data.taskId);
        } else {
            showTrainingStatus(data.error || '开始训练失败', 'error');
        }
    } catch (error) {
        showTrainingStatus('网络错误，请稍后重试', 'error');
    }
}

// 轮询训练状态
async function pollTrainingStatus(taskId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/model/status/${taskId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            const data = await response.json();
            if (response.ok) {
                showTrainingStatus(`训练进度: ${data.progress}%`, 'info');

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    showTrainingStatus('模型训练完成！', 'success');
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    showTrainingStatus('模型训练失败: ' + data.error, 'error');
                }
            }
        } catch (error) {
            console.error('获取训练状态失败:', error);
        }
    }, 5000); // 每5秒轮询一次
}

// 语音克隆功能
document.addEventListener('DOMContentLoaded', function () {
    // 页面加载时初始化语音克隆功能
    initSynthesisFeatures();

    // 加载用户的模型列表
    loadUserModels();
});

// 初始化语音合成功能
function initSynthesisFeatures() {
    // 滑块值实时显示
    const speedControl = document.getElementById('speed-control');
    const pitchControl = document.getElementById('pitch-control');

    if (speedControl && pitchControl) {
        const speedValue = speedControl.parentElement.querySelector('.slider-value');
        const pitchValue = pitchControl.parentElement.querySelector('.slider-value');

        speedControl.addEventListener('input', () => {
            speedValue.textContent = parseFloat(speedControl.value).toFixed(1);
        });

        pitchControl.addEventListener('input', () => {
            pitchValue.textContent = parseFloat(pitchControl.value).toFixed(1);
        });

        // 开始合成按钮点击事件
        const startSynthesisBtn = document.getElementById('start-synthesis');
        if (startSynthesisBtn) {
            startSynthesisBtn.addEventListener('click', startSynthesis);
        }

        // 下载和分享按钮
        const downloadBtn = document.getElementById('download-synthesis');
        const shareBtn = document.getElementById('share-synthesis');

        if (downloadBtn) {
            downloadBtn.addEventListener('click', downloadSynthesizedAudio);
        }

        if (shareBtn) {
            shareBtn.addEventListener('click', shareSynthesizedAudio);
        }
    }
}

// 加载用户的模型列表
async function loadUserModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/list`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            displayUserModels(data.models || []);
        } else {
            console.error('Failed to load models');
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// 显示用户的模型列表
function displayUserModels(models) {
    const modelsContainer = document.getElementById('models-list');
    if (!modelsContainer) return;

    // 保留默认模型
    const defaultModel = modelsContainer.innerHTML;

    // 添加用户模型
    models.forEach(model => {
        const date = new Date(model.createdAt);
        const formattedDate = date.toLocaleDateString('zh-CN');

        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        modelCard.dataset.modelId = model.id;
        modelCard.innerHTML = `
            <div class="model-icon">🤖</div>
            <div class="model-info">
                <div class="model-name">${model.name}</div>
                <div class="model-date">创建于 ${formattedDate}</div>
            </div>
        `;

        // 点击选择模型
        modelCard.addEventListener('click', () => {
            document.querySelectorAll('.model-card').forEach(card => {
                card.classList.remove('active');
            });
            modelCard.classList.add('active');
        });

        modelsContainer.appendChild(modelCard);
    });

    // 如果没有模型，显示提示
    if (models.length === 0) {
        const noModels = document.createElement('div');
        noModels.className = 'no-models';
        noModels.innerHTML = `
            <div class="placeholder-icon">🎭</div>
            <div class="placeholder-text">您还没有训练好的模型</div>
            <small>请先在"模型训练"标签页训练您的语音模型</small>
        `;

        modelsContainer.appendChild(noModels);
    }
}

// 开始语音合成
async function startSynthesis() {
    const textToSynthesize = document.getElementById('synthesis-text').value.trim();
    if (!textToSynthesize) {
        showSynthesisMessage('请输入要合成的文本', 'error');
        return;
    }

    // 获取选中的模型
    const selectedModel = document.querySelector('.model-card.active');
    if (!selectedModel) {
        showSynthesisMessage('请选择一个模型', 'error');
        return;
    }

    const modelId = selectedModel.dataset.modelId || 'default';

    // 获取语速和音调
    const speed = parseFloat(document.getElementById('speed-control').value);
    const pitch = parseFloat(document.getElementById('pitch-control').value);

    // 显示合成中的状态
    showSynthesisLoading();

    try {
        // 这里添加实际的API调用
        // 模拟API调用和响应
        setTimeout(() => {
            // 模拟成功响应
            showSynthesisResult();

            // 添加到历史记录
            addToSynthesisHistory(textToSynthesize, modelId);
        }, 2000);

        /* 实际API调用示例：
        const response = await fetch(`${API_BASE_URL}/synthesis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                text: textToSynthesize,
                modelId,
                params: {
                    speed,
                    pitch
                }
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            showSynthesisResult(data.audioUrl);
            addToSynthesisHistory(textToSynthesize, modelId, data.audioUrl);
        } else {
            const error = await response.json();
            showSynthesisMessage(error.message || '合成失败', 'error');
        }
        */

    } catch (error) {
        showSynthesisMessage('网络错误，请稍后重试', 'error');
        console.error('Synthesis error:', error);
    }
}

// 显示合成中的加载状态
function showSynthesisLoading() {
    const placeholder = document.getElementById('synthesis-placeholder');
    const result = document.getElementById('synthesis-result');

    if (placeholder && result) {
        placeholder.style.display = 'flex';
        result.style.display = 'none';

        placeholder.innerHTML = `
            <div class="live-voice-indicator">🎤</div>
            <div class="placeholder-icon">
                <div class="loading"></div>
            </div>
            <div class="placeholder-text">正在合成语音，请稍候...</div>
        `;
    }
}

// 显示合成结果
function showSynthesisResult(audioUrl = null) {
    const placeholder = document.getElementById('synthesis-placeholder');
    const result = document.getElementById('synthesis-result');
    const audio = document.getElementById('synthesis-audio');

    if (placeholder && result && audio) {
        placeholder.style.display = 'none';
        result.style.display = 'block';

        // 设置音频源
        // 如果没有真实URL，使用示例音频
        audio.src = audioUrl || 'https://example.com/sample-audio.mp3';

        // 在真实环境中，会设置为API返回的音频URL
        // audio.src = audioUrl;

        // 模拟波形可视化
        const waveVisualization = document.querySelector('.wave-visualization');
        if (waveVisualization) {
            // 实际应用中，可以基于音频数据创建真实的波形图
            waveVisualization.style.animationPlayState = 'running';
        }
    }
}

// 显示合成消息
function showSynthesisMessage(message, type = 'info') {
    // 创建消息元素
    const messageElement = document.createElement('div');
    messageElement.className = `synthesis-message ${type}`;
    messageElement.textContent = message;

    // 添加到合成区域
    const container = document.querySelector('.synthesis-container');
    if (container) {
        container.appendChild(messageElement);

        // 3秒后移除消息
        setTimeout(() => {
            messageElement.classList.add('fade-out');
            setTimeout(() => {
                messageElement.remove();
            }, 300);
        }, 3000);
    }
}

// 添加到合成历史记录
function addToSynthesisHistory(text, modelId, audioUrl = null) {
    const historyList = document.getElementById('synthesis-history-list');
    if (!historyList) return;

    const now = new Date();
    const formattedDate = now.toLocaleString('zh-CN');

    // 获取模型名称
    let modelName = '默认模型';
    const modelCard = document.querySelector(`.model-card[data-model-id="${modelId}"]`);
    if (modelCard) {
        modelName = modelCard.querySelector('.model-name').textContent;
    }

    // 创建历史记录项
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.innerHTML = `
        <div class="history-time">${formattedDate}</div>
        <div class="history-content">
            <div class="history-text">${text.length > 50 ? text.substring(0, 50) + '...' : text}</div>
            <div class="history-model">使用模型: ${modelName}</div>
        </div>
        <div class="history-actions">
            <button class="history-play" data-audio="${audioUrl || ''}">
                <span>播放</span>
            </button>
        </div>
    `;

    // 添加到历史列表的顶部
    if (historyList.firstChild) {
        historyList.insertBefore(historyItem, historyList.firstChild);
    } else {
        historyList.appendChild(historyItem);
    }

    // 添加播放功能
    const playButton = historyItem.querySelector('.history-play');
    if (playButton) {
        playButton.addEventListener('click', () => {
            const audio = document.getElementById('synthesis-audio');
            if (audio) {
                // 在实际应用中，使用真实的音频URL
                // audio.src = playButton.dataset.audio;
                audio.play();
            }
        });
    }
}

// 下载合成的音频
function downloadSynthesizedAudio() {
    const audio = document.getElementById('synthesis-audio');
    if (audio && audio.src) {
        // 创建一个临时链接来下载音频
        const link = document.createElement('a');
        link.href = audio.src;
        link.download = `synthesized_audio_${Date.now()}.mp3`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        showSynthesisMessage('音频下载已开始', 'success');
    } else {
        showSynthesisMessage('没有可下载的音频', 'error');
    }
}

// 分享合成的音频
function shareSynthesizedAudio() {
    const text = document.getElementById('synthesis-text').value.trim();

    // 检查Web Share API是否可用
    if (navigator.share) {
        navigator.share({
            title: '我用EasyClone合成的语音',
            text: `听听我用AI合成的语音: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`,
            // url: window.location.href
        })
            .then(() => {
                showSynthesisMessage('分享成功', 'success');
            })
            .catch(error => {
                console.error('Share error:', error);
                showSynthesisMessage('分享失败', 'error');
            });
    } else {
        // Web Share API不可用，显示提示
        const audio = document.getElementById('synthesis-audio');
        if (audio && audio.src) {
            // 将音频URL复制到剪贴板
            const textArea = document.createElement('textarea');
            textArea.value = audio.src;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            showSynthesisMessage('音频链接已复制到剪贴板', 'success');
        } else {
            showSynthesisMessage('没有可分享的音频', 'error');
        }
    }
} 