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

    // 清空容器
    modelsContainer.innerHTML = '';

    // 添加系统内置的默认模型
    const defaultModelCard = document.createElement('div');
    defaultModelCard.className = 'model-card active'; // 默认选中
    defaultModelCard.dataset.modelId = 'default';
    defaultModelCard.innerHTML = `
        <div class="model-icon">🔊</div>
        <div class="model-info">
            <div class="model-name">系统内置-默认模型</div>
            <div class="model-date">系统默认 | <span style="color: #3498db;">通用模型</span></div>
        </div>
    `;

    // 点击选择模型
    defaultModelCard.addEventListener('click', () => {
        document.querySelectorAll('.model-card').forEach(card => {
            card.classList.remove('active');
        });
        defaultModelCard.classList.add('active');
    });

    modelsContainer.appendChild(defaultModelCard);

    // 添加用户模型
    models.forEach(model => {
        const date = new Date(model.createdAt);
        const formattedDate = date.toLocaleDateString('zh-CN');

        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        modelCard.dataset.modelId = model.mid;
        modelCard.innerHTML = `
            <div class="model-icon">🤖</div>
            <div class="model-info">
                <div class="model-name">${model.modelName}</div>
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

    // 如果没有用户模型，不显示"没有模型"提示，因为我们已经添加了默认模型
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
    const modelName = selectedModel.querySelector('.model-name').textContent;

    // 获取语速和音调
    const speed = parseFloat(document.getElementById('speed-control').value);
    const pitch = parseFloat(document.getElementById('pitch-control').value);

    // 显示合成中的状态
    showSynthesisLoading();

    try {
        // 模拟API调用和响应
        setTimeout(() => {
            // 模拟成功响应
            // 根据不同的模型使用不同的示例音频
            let audioUrl = null;

            if (modelId === 'thchs30_1') {
                // 使用THCHS30模型的示例音频文件
                audioUrl = './assets/demo_thchs30.mp3';
                showSynthesisMessage(`使用 ${modelName} 模型合成成功！`, 'success');
            } else if (modelId === 'default') {
                // 使用默认模型的示例音频文件
                audioUrl = './assets/demo_default.mp3';
                showSynthesisMessage(`使用默认模型合成成功！`, 'success');
            } else {
                // 其他模型
                audioUrl = './assets/demo_audio.mp3';
                showSynthesisMessage(`使用 ${modelName} 模型合成成功！`, 'success');
            }

            showSynthesisResult(audioUrl);

            // 添加到历史记录
            addToSynthesisHistory(textToSynthesize, modelId, audioUrl);
        }, 2000);

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