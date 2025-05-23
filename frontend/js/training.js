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

        // 添加点击事件处理
        const checkbox = audioElement.querySelector(`input[type="checkbox"]`);
        const customCheckbox = audioElement.querySelector('.checkbox-custom');
        const audioInfo = audioElement.querySelector('.audio-info');

        // 自定义复选框点击事件
        customCheckbox.addEventListener('click', () => {
            checkbox.checked = !checkbox.checked;
        });

        // 点击音频信息区域也可以选择
        audioInfo.addEventListener('click', () => {
            checkbox.checked = !checkbox.checked;
        });
    });
}

// 显示训练状态
function showTrainingStatus(message, type = 'info') {
    const statusDiv = document.getElementById('training-status');
    if (!statusDiv) return;

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

// 初始化训练页面
document.addEventListener('DOMContentLoaded', function () {
    const trainButton = document.getElementById('start-training-btn');
    if (trainButton) {
        trainButton.addEventListener('click', startTraining);
    }
}); 