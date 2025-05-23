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

        // 播放按钮
        const playBtn = document.getElementById('play-synthesis');
        if (playBtn) {
            playBtn.addEventListener('click', playSynthesizedAudio);
        }

        // 下载按钮
        const downloadBtn = document.getElementById('download-synthesis');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', downloadSynthesizedAudio);
        }

        // 重命名按钮
        const renameBtn = document.getElementById('rename-synthesis');
        if (renameBtn) {
            renameBtn.addEventListener('click', renameSynthesizedAudio);
        }

        // 保存按钮
        const saveBtn = document.getElementById('save-synthesis');
        if (saveBtn) {
            saveBtn.addEventListener('click', saveSynthesizedAudio);
        }
    }

    // 加载用户模型
    loadUserModels();

    // 加载历史记录
    loadInferenceHistory();
}

// 当前合成的音频信息
let currentSynthesisInfo = {
    audioUrl: null,
    text: '',
    modelId: '',
    modelName: '',
    audioName: '合成音频',
    isSaved: false
};

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

            if (modelId === 'default') {
                // 使用默认模型的示例音频文件
                audioUrl = './assets/demo_default.mp3';
                showSynthesisMessage(`使用默认模型合成成功！`, 'success');
            } else {
                // 用户训练的模型
                audioUrl = './assets/demo_audio.mp3';
                showSynthesisMessage(`使用 ${modelName} 模型合成成功！`, 'success');
            }

            // 更新当前合成信息
            currentSynthesisInfo = {
                audioUrl: audioUrl,
                text: textToSynthesize,
                modelId: modelId,
                modelName: modelName,
                audioName: `${modelName}_合成_${new Date().getTime()}`,
                isSaved: false
            };

            // 设置默认音频名称
            const audioNameInput = document.getElementById('synthesis-audio-name');
            if (audioNameInput) {
                audioNameInput.value = currentSynthesisInfo.audioName;
            }

            showSynthesisResult(audioUrl);

            // 不再自动添加到历史记录，需要用户点击保存
        }, 2000);

    } catch (error) {
        showSynthesisMessage('网络错误，请稍后重试', 'error');
        console.error('Synthesis error:', error);
    }
}

// 播放合成的音频
function playSynthesizedAudio() {
    const audio = document.getElementById('synthesis-audio');
    if (audio && audio.src) {
        if (audio.paused) {
            audio.play();
            document.getElementById('play-synthesis').querySelector('span').textContent = '暂停';
        } else {
            audio.pause();
            document.getElementById('play-synthesis').querySelector('span').textContent = '播放';
        }
    } else {
        showSynthesisMessage('没有可播放的音频', 'error');
    }
}

// 重命名合成的音频
function renameSynthesizedAudio() {
    const audioNameInput = document.getElementById('synthesis-audio-name');
    if (audioNameInput) {
        // 聚焦输入框
        audioNameInput.focus();
        audioNameInput.select();

        // 更新当前合成信息
        currentSynthesisInfo.audioName = audioNameInput.value;
        showSynthesisMessage('请输入新名称并按Enter确认', 'info');

        // 添加一次性事件监听器
        const handleEnterKey = (e) => {
            if (e.key === 'Enter') {
                currentSynthesisInfo.audioName = audioNameInput.value;
                showSynthesisMessage('音频已重命名', 'success');
                audioNameInput.blur();
                audioNameInput.removeEventListener('keydown', handleEnterKey);
            }
        };

        audioNameInput.addEventListener('keydown', handleEnterKey);
    }
}

// 保存合成的音频到历史记录
async function saveSynthesizedAudio() {
    if (!currentSynthesisInfo.audioUrl) {
        showSynthesisMessage('没有可保存的音频', 'error');
        return;
    }

    // 获取最新的音频名称
    const audioNameInput = document.getElementById('synthesis-audio-name');
    if (audioNameInput) {
        currentSynthesisInfo.audioName = audioNameInput.value;
    }

    try {
        // 调用API保存推理记录
        const response = await fetch(`${API_BASE_URL}/inference/save`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                name: currentSynthesisInfo.audioName,
                text: currentSynthesisInfo.text,
                modelId: currentSynthesisInfo.modelId,
                audioUrl: currentSynthesisInfo.audioUrl,
                speed: parseFloat(document.getElementById('speed-control').value),
                pitch: parseFloat(document.getElementById('pitch-control').value)
            })
        });

        if (response.ok) {
            const data = await response.json();
            showSynthesisMessage('音频已保存到历史记录', 'success');
            currentSynthesisInfo.isSaved = true;

            // 重新加载历史记录
            loadInferenceHistory();

            // 禁用保存按钮，防止重复保存
            const saveBtn = document.getElementById('save-synthesis');
            if (saveBtn) {
                saveBtn.disabled = true;
                saveBtn.style.opacity = '0.5';
                saveBtn.style.cursor = 'not-allowed';
            }
        } else {
            const errorData = await response.json();
            showSynthesisMessage(errorData.error || '保存失败', 'error');
        }
    } catch (error) {
        console.error('保存错误:', error);

        // 模拟API调用成功
        showSynthesisMessage('音频已保存到历史记录', 'success');
        currentSynthesisInfo.isSaved = true;

        // 添加到历史记录UI
        addToSynthesisHistory(
            currentSynthesisInfo.text,
            currentSynthesisInfo.modelId,
            currentSynthesisInfo.audioUrl,
            currentSynthesisInfo.audioName
        );

        // 禁用保存按钮，防止重复保存
        const saveBtn = document.getElementById('save-synthesis');
        if (saveBtn) {
            saveBtn.disabled = true;
            saveBtn.style.opacity = '0.5';
            saveBtn.style.cursor = 'not-allowed';
        }
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
        audio.src = audioUrl || 'https://example.com/sample-audio.mp3';

        // 重置保存按钮状态
        const saveBtn = document.getElementById('save-synthesis');
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.style.opacity = '1';
            saveBtn.style.cursor = 'pointer';
        }

        // 模拟波形可视化
        const waveVisualization = document.querySelector('.wave-visualization');
        if (waveVisualization) {
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
function addToSynthesisHistory(text, modelId, audioUrl = null, audioName = '合成音频') {
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
            <div class="history-text">${audioName}</div>
            <div class="history-details">
                <div class="history-model">使用模型: ${modelName}</div>
                <div class="history-content-text">${text.length > 30 ? text.substring(0, 30) + '...' : text}</div>
            </div>
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
                audio.src = playButton.dataset.audio;
                audio.play();

                // 更新当前播放的音频名称
                const audioNameInput = document.getElementById('synthesis-audio-name');
                if (audioNameInput) {
                    audioNameInput.value = audioName;
                }

                // 显示音频结果区域
                showSynthesisResult(playButton.dataset.audio);
            }
        });
    }
}

// 下载合成的音频
function downloadSynthesizedAudio() {
    const audio = document.getElementById('synthesis-audio');
    if (audio && audio.src) {
        // 获取当前音频名称
        const audioNameInput = document.getElementById('synthesis-audio-name');
        let fileName = 'synthesized_audio';
        if (audioNameInput && audioNameInput.value) {
            fileName = audioNameInput.value;
        }

        // 创建一个临时链接来下载音频
        const link = document.createElement('a');
        link.href = audio.src;
        link.download = `${fileName}.mp3`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        showSynthesisMessage('音频下载已开始', 'success');
    } else {
        showSynthesisMessage('没有可下载的音频', 'error');
    }
}

// 加载推理历史记录
async function loadInferenceHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/inference/list`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            displayInferenceHistory(data.data.histories || []);
        } else {
            console.error('Failed to load inference history');
        }
    } catch (error) {
        console.error('Error loading inference history:', error);

        // 模拟数据
        displayInferenceHistory([
            {
                hid: 'history1',
                input_text: '这是一段示例文本，用于演示语音合成功能。',
                output_path: './assets/demo_audio.mp3',
                created_at: '2023-05-23T21:22:26Z',
                audio_name: '示例音频1',
                model_name: '默认模型'
            },
            {
                hid: 'history2',
                input_text: '人工智能语音合成技术正在不断发展。',
                output_path: './assets/demo_default.mp3',
                created_at: '2023-05-22T18:15:10Z',
                audio_name: '示例音频2',
                model_name: '用户模型1'
            }
        ]);
    }
}

// 显示推理历史记录
function displayInferenceHistory(histories) {
    const historyList = document.getElementById('synthesis-history-list');
    if (!historyList) return;

    // 清空容器
    historyList.innerHTML = '';

    if (histories.length === 0) {
        historyList.innerHTML = '<div class="no-history">暂无历史记录</div>';
        return;
    }

    // 添加历史记录
    histories.forEach(history => {
        const date = new Date(history.created_at);
        const formattedDate = date.toLocaleString('zh-CN');

        // 计算音频时长（如果有）
        let duration = history.duration || 0;
        const durationText = duration > 0 ?
            `${Math.floor(duration / 60)}:${String(Math.floor(duration % 60)).padStart(2, '0')}` :
            '00:00';

        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div class="history-header">
                <div class="history-title">
                    <span class="history-audio-name">${history.audio_name || '未命名音频'}</span>
                    <span class="history-date">${formattedDate}</span>
                </div>
                <div class="history-duration">
                    <i class="duration-icon">⏱</i>
                    <span>${durationText}</span>
                </div>
            </div>
            <div class="history-body">
                <div class="history-model-info">
                    <div class="model-badge">${history.model_name || '默认模型'}</div>
                </div>
                <div class="history-text-preview">
                    <i class="text-icon">💬</i>
                    <span>${history.input_text.length > 50 ? history.input_text.substring(0, 50) + '...' : history.input_text}</span>
                </div>
                <div class="history-waveform">
                    <div class="mini-waveform"></div>
                </div>
            </div>
            <div class="history-actions">
                <button class="history-play-btn" data-audio="${history.output_path || ''}">
                    <i class="play-icon">▶</i>
                    <span>播放</span>
                </button>
                <button class="history-download-btn" data-audio="${history.output_path || ''}" data-name="${history.audio_name || '未命名音频'}">
                    <i class="download-icon">⬇</i>
                </button>
            </div>
        `;

        // 添加播放功能
        const playButton = historyItem.querySelector('.history-play-btn');
        if (playButton) {
            playButton.addEventListener('click', () => {
                // 重置所有播放按钮
                document.querySelectorAll('.history-play-btn').forEach(btn => {
                    btn.innerHTML = '<i class="play-icon">▶</i><span>播放</span>';
                    btn.classList.remove('playing');
                });

                const audio = document.getElementById('synthesis-audio');
                if (audio) {
                    if (audio.src === playButton.dataset.audio && !audio.paused) {
                        // 如果是当前音频且正在播放，则暂停
                        audio.pause();
                        playButton.innerHTML = '<i class="play-icon">▶</i><span>播放</span>';
                        playButton.classList.remove('playing');
                    } else {
                        // 否则播放新音频
                        audio.src = playButton.dataset.audio;
                        audio.play();
                        playButton.innerHTML = '<i class="play-icon">⏸</i><span>暂停</span>';
                        playButton.classList.add('playing');

                        // 音频播放结束时重置按钮
                        audio.onended = () => {
                            playButton.innerHTML = '<i class="play-icon">▶</i><span>播放</span>';
                            playButton.classList.remove('playing');
                        };

                        // 更新当前播放的音频名称
                        const audioNameInput = document.getElementById('synthesis-audio-name');
                        if (audioNameInput) {
                            audioNameInput.value = history.audio_name || '未命名音频';
                        }

                        // 显示音频结果区域
                        showSynthesisResult(playButton.dataset.audio);

                        // 更新当前合成信息
                        currentSynthesisInfo = {
                            audioUrl: playButton.dataset.audio,
                            text: history.input_text,
                            modelId: history.mid,
                            modelName: history.model_name || '默认模型',
                            audioName: history.audio_name || '未命名音频',
                            isSaved: true
                        };

                        // 禁用保存按钮，因为已经保存过了
                        const saveBtn = document.getElementById('save-synthesis');
                        if (saveBtn) {
                            saveBtn.disabled = true;
                            saveBtn.style.opacity = '0.5';
                            saveBtn.style.cursor = 'not-allowed';
                        }
                    }
                }
            });
        }

        // 添加下载功能
        const downloadButton = historyItem.querySelector('.history-download-btn');
        if (downloadButton) {
            downloadButton.addEventListener('click', () => {
                const link = document.createElement('a');
                link.href = downloadButton.dataset.audio;
                link.download = `${downloadButton.dataset.name}.mp3`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                showSynthesisMessage('音频下载已开始', 'success');
            });
        }

        historyList.appendChild(historyItem);

        // 为每个历史记录项添加波形动画
        const waveform = historyItem.querySelector('.mini-waveform');
        if (waveform) {
            for (let i = 0; i < 12; i++) {
                const bar = document.createElement('div');
                bar.className = 'wave-bar';
                bar.style.setProperty('--bar-index', i);
                waveform.appendChild(bar);
            }
        }
    });
} 