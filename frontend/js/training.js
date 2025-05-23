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

// 加载用户模型列表
async function loadUserModelsForTraining() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/list`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            displayUserModelsForTraining(data.models || []);
        } else {
            console.error('Failed to load models for training');
        }
    } catch (error) {
        console.error('Error loading models for training:', error);
    }
}

// 显示用户模型列表（训练中和已完成）
function displayUserModelsForTraining(models) {
    const trainingModelsList = document.getElementById('training-models-list');
    const completedModelsList = document.getElementById('completed-models-list');

    if (!trainingModelsList || !completedModelsList) return;

    // 清空容器
    trainingModelsList.innerHTML = '';
    completedModelsList.innerHTML = '';

    // 分类模型
    const trainingModels = [];
    const completedModels = [];

    // 分类用户模型
    models.forEach(model => {
        if (model.state === 1) {
            trainingModels.push(model);
        } else if (model.state === 2) {
            completedModels.push(model);
        }
    });

    // 显示训练中的模型
    if (trainingModels.length === 0) {
        trainingModelsList.innerHTML = '<div class="no-models-message">暂无训练中的模型</div>';
    } else {
        trainingModels.forEach(model => {
            const modelElement = createModelElement(model, 'in-progress');
            trainingModelsList.appendChild(modelElement);
        });
    }

    // 显示已完成的模型
    if (completedModels.length === 0) {
        completedModelsList.innerHTML = '<div class="no-models-message">暂无已完成的模型</div>';
    } else {
        completedModels.forEach(model => {
            const modelElement = createModelElement(model, 'completed');
            completedModelsList.appendChild(modelElement);
        });
    }
}

// 创建模型元素
function createModelElement(model, status) {
    const date = new Date(model.createdAt);
    const formattedDate = date.toLocaleDateString('zh-CN');

    const modelElement = document.createElement('div');
    modelElement.className = `model-item ${status}`;
    modelElement.dataset.modelId = model.mid;

    let statusText = status === 'in-progress' ? '训练中' : '已完成';
    let statusClass = status === 'in-progress' ? 'in-progress' : 'completed';
    let statusColor = status === 'in-progress' ? '#e74c3c' : '#2ecc71';

    modelElement.innerHTML = `
        <div class="model-item-name">${model.modelName}</div>
        <div class="model-item-status">
            <span class="status-badge ${statusClass}"></span>
            <span style="color: ${statusColor}">${statusText}</span>
        </div>
        <div class="model-item-date">创建于 ${formattedDate}</div>
    `;

    // 点击事件 - 选择模型
    modelElement.addEventListener('click', () => {
        // 如果是已完成的模型，可以选择它
        if (status === 'completed') {
            document.querySelectorAll('.model-item').forEach(item => {
                item.classList.remove('active');
            });
            modelElement.classList.add('active');

            // 可以在这里添加选中模型后的逻辑
            console.log(`选中模型: ${model.modelName}, ID: ${model.mid}`);
        }
    });

    return modelElement;
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

            // 刷新模型列表
            setTimeout(() => {
                loadUserModelsForTraining();
            }, 2000);
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

                    // 刷新模型列表
                    loadUserModelsForTraining();
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
    // 加载用户模型
    loadUserModelsForTraining();

    // 绑定开始训练按钮事件
    const trainButton = document.getElementById('start-training-btn');
    if (trainButton) {
        trainButton.addEventListener('click', startTraining);
    }

    // 切换到训练页面时刷新模型列表
    document.querySelectorAll('.sidebar li').forEach(item => {
        item.addEventListener('click', function () {
            if (this.textContent === '模型训练') {
                loadUserModelsForTraining();
            }
        });
    });
}); 