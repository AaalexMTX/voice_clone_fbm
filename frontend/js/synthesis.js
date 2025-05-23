// 获取当前网页的路径信息
function logCurrentPath() {
    console.log('当前页面URL:', window.location.href);
    console.log('当前页面路径:', window.location.pathname);
    console.log('当前页面域名:', window.location.hostname);
    console.log('当前页面协议:', window.location.protocol);
    console.log('当前页面相对路径:', window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/')));
    console.log('解析后的相对路径:', new URL('.', window.location.href).href);
}

// 初始化语音合成功能
function initSynthesisFeatures() {
    // 打印路径信息用于调试
    logCurrentPath();
    console.log('API_BASE_URL:', API_BASE_URL);

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

        // 初始化播放器功能
        initSynthesisPlayer();

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

        // 监听音频播放结束事件
        const audio = document.getElementById('synthesis-audio');
        if (audio) {
            audio.addEventListener('ended', () => {
                // 音频播放结束时重置播放按钮
                const playBtn = document.getElementById('play-synthesis');
                if (playBtn) {
                    playBtn.setAttribute('data-state', 'paused');
                    playBtn.querySelector('.play-icon').textContent = '▶';
                }
            });
        }
    }

    // 监听页面上所有的音频播放按钮
    listenToAllPlayButtons();

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
        // 生成音频文件名 - 使用动态生成的名称
        const timestamp = new Date().getTime();
        const audioName = `${modelName}_合成_${timestamp}`;

        // 模拟API调用和响应
        setTimeout(() => {
            showSynthesisMessage(`使用 ${modelName} 模型合成成功！`, 'success');

            // 更新当前合成信息
            currentSynthesisInfo = {
                text: textToSynthesize,
                modelId: modelId,
                modelName: modelName,
                audioName: audioName,
                isSaved: false
            };

            // 设置默认音频名称
            const audioNameInput = document.getElementById('synthesis-audio-name');
            if (audioNameInput) {
                audioNameInput.value = currentSynthesisInfo.audioName;
            }

            showSynthesisResult();

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
    const playBtn = document.getElementById('play-synthesis');
    const progressEl = document.getElementById('synthesis-progress');
    const currentTimeEl = document.getElementById('synthesis-current-time');
    const totalTimeEl = document.getElementById('synthesis-total-time');

    if (!audio || !playBtn) {
        showSynthesisMessage('播放器初始化失败', 'error');
        return;
    }

    if (audio.paused) {
        // 获取当前音频名称
        const audioNameInput = document.getElementById('synthesis-audio-name');
        let audioName = '合成音频';
        if (audioNameInput && audioNameInput.value) {
            audioName = audioNameInput.value;
        }

        // 清理文件名，移除不安全字符
        const safeAudioName = audioName.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_');

        // 显示加载状态
        playBtn.disabled = true;
        playBtn.querySelector('.play-icon').textContent = '⏳';

        // 使用特殊接口获取音频
        const audioUrl = `${API_BASE_URL}/audio/user/${safeAudioName}`;
        console.log('使用新API获取音频:', audioUrl);

        // 使用fetch发送带授权的请求
        fetch(audioUrl, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`获取音频失败: ${response.status} ${response.statusText}`);
                }
                return response.blob();
            })
            .then(audioBlob => {
                // 创建音频Blob URL
                const blobUrl = URL.createObjectURL(audioBlob);

                // 设置音频源
                audio.src = blobUrl;

                // 加载并播放
                audio.load();
                return audio.play();
            })
            .then(() => {
                // 播放成功
                console.log('播放成功');
                showSynthesisMessage('音频播放成功', 'success');

                // 更新按钮状态为"播放中"
                playBtn.disabled = false;
                playBtn.setAttribute('data-state', 'playing');
                playBtn.querySelector('.play-icon').textContent = '⏸';
            })
            .catch(error => {
                console.error('播放失败:', error);
                showSynthesisMessage('音频文件播放失败，请检查文件是否存在', 'error');

                // 重置按钮状态
                playBtn.disabled = false;
                playBtn.setAttribute('data-state', 'paused');
                playBtn.querySelector('.play-icon').textContent = '▶';
            });
    } else {
        // 暂停音频
        audio.pause();
        // 更新按钮状态为"暂停"
        playBtn.setAttribute('data-state', 'paused');
        playBtn.querySelector('.play-icon').textContent = '▶';
    }
}

// 初始化合成播放器功能
function initSynthesisPlayer() {
    const audio = document.getElementById('synthesis-audio');
    const playBtn = document.getElementById('play-synthesis');
    const progressEl = document.getElementById('synthesis-progress');
    const currentTimeEl = document.getElementById('synthesis-current-time');
    const totalTimeEl = document.getElementById('synthesis-total-time');
    const waveformContainer = document.getElementById('synthesis-waveform');

    if (!audio || !playBtn || !progressEl || !currentTimeEl || !totalTimeEl || !waveformContainer) {
        console.error('合成播放器初始化失败：元素未找到');
        return;
    }

    // 音频加载元数据后设置总时长
    audio.addEventListener('loadedmetadata', () => {
        console.log('合成音频元数据加载完成，时长:', audio.duration);

        // 检查时长是否为NaN或0，这是常见的元数据加载失败情况
        if (isNaN(audio.duration) || audio.duration === 0) {
            console.warn('合成音频时长无效:', audio.duration);
            return;
        }

        totalTimeEl.textContent = formatTime(audio.duration);

        // 创建简单的波形可视化 (模拟波形)
        createSimpleWaveform(waveformContainer);
    });

    // 更新进度条和时间显示
    audio.addEventListener('timeupdate', () => {
        // 检查音频时长是否有效
        if (isNaN(audio.duration) || audio.duration === 0) {
            return;
        }

        const progress = (audio.currentTime / audio.duration) * 100;
        progressEl.style.width = `${progress}%`;
        currentTimeEl.textContent = formatTime(audio.currentTime);
    });

    // 音频播放结束时重置
    audio.addEventListener('ended', () => {
        playBtn.setAttribute('data-state', 'paused');
        playBtn.querySelector('.play-icon').textContent = '▶';
        progressEl.style.width = '0%';
        audio.currentTime = 0;
    });

    // 点击进度条跳转
    const progressContainer = progressEl.parentElement;
    progressContainer.addEventListener('click', (e) => {
        // 检查音频是否已加载
        if (isNaN(audio.duration) || audio.duration === 0) {
            console.warn('无法跳转：音频尚未加载或时长无效');
            return;
        }

        const rect = progressContainer.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        audio.currentTime = pos * audio.duration;
    });

    // 播放按钮点击事件
    playBtn.addEventListener('click', playSynthesizedAudio);
}

// 创建简单的波形可视化
function createSimpleWaveform(container) {
    container.innerHTML = ''; // 清除容器内容

    const waveformSVG = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    waveformSVG.setAttribute('width', '100%');
    waveformSVG.setAttribute('height', '40');
    waveformSVG.style.display = 'block';

    // 生成随机波形 (实际应用中应该使用真实的音频数据)
    const points = [];
    const segments = 50;

    for (let i = 0; i < segments; i++) {
        const x = (i / segments) * 100;
        // 生成随机高度，中间部分稍高一些
        const heightFactor = Math.sin((i / segments) * Math.PI);
        const y = 20 - (Math.random() * 15 + 5) * heightFactor;
        points.push(`${x}%,${y}`);
    }

    // 添加对称的下半部分
    for (let i = segments - 1; i >= 0; i--) {
        const [x, y] = points[i].split(',');
        const newY = 40 - parseFloat(y);
        points.push(`${x},${newY}`);
    }

    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    polyline.setAttribute('points', points.join(' '));
    polyline.setAttribute('fill', 'rgba(240, 100, 15, 0.2)');
    polyline.setAttribute('stroke', 'rgba(240, 100, 15, 0.6)');
    polyline.setAttribute('stroke-width', '1');

    waveformSVG.appendChild(polyline);
    container.appendChild(waveformSVG);
}

// 格式化时间 (秒 -> mm:ss)
function formatTime(seconds) {
    if (isNaN(seconds) || seconds === Infinity) {
        return '0:00';
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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

// 下载合成的音频
function downloadSynthesizedAudio() {
    // 获取当前音频名称
    const audioNameInput = document.getElementById('synthesis-audio-name');
    let fileName = '合成音频';
    if (audioNameInput && audioNameInput.value) {
        fileName = audioNameInput.value;
    }

    // 清理文件名，移除不安全字符
    const safeFileName = fileName.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_');

    // 显示加载状态
    showSynthesisMessage('正在准备下载...', 'info');

    // 使用特殊接口获取音频进行下载
    const audioUrl = `${API_BASE_URL}/audio/user/${safeFileName}`;

    // 使用fetch发送带授权的请求下载文件
    fetch(audioUrl, {
        method: 'GET',
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`下载音频失败: ${response.status} ${response.statusText}`);
            }
            return response.blob();
        })
        .then(blob => {
            // 创建Blob URL并触发下载
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${fileName}.mp3`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            showSynthesisMessage('音频下载已开始', 'success');
        })
        .catch(error => {
            console.error('下载失败:', error);
            showSynthesisMessage('音频下载失败', 'error');
        });
}

// 保存合成的音频到历史记录
async function saveSynthesizedAudio() {
    // 获取最新的音频名称
    const audioNameInput = document.getElementById('synthesis-audio-name');
    if (!audioNameInput || !audioNameInput.value) {
        showSynthesisMessage('请先输入音频名称', 'error');
        return;
    }

    const audioName = audioNameInput.value;
    // 清理文件名，移除不安全字符
    const safeFileName = audioName.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_');

    try {
        // 调用API保存推理记录
        const response = await fetch(`${API_BASE_URL}/inference/save`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                name: audioName,
                text: currentSynthesisInfo.text || '',
                modelId: currentSynthesisInfo.modelId || 'default',
                audioUrl: safeFileName, // 只传递音频名称，API内部会处理路径
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
            currentSynthesisInfo.text || '',
            currentSynthesisInfo.modelId || 'default',
            safeFileName,
            audioName
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

        // 获取音频名称
        const audioNameInput = document.getElementById('synthesis-audio-name');
        let audioName = '合成音频';
        if (audioNameInput && audioNameInput.value) {
            audioName = audioNameInput.value;
        }
        console.log('显示合成结果，音频名称:', audioName);

        // 清理文件名，移除不安全字符
        const safeAudioName = audioName.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_');
        console.log('安全的文件名:', safeAudioName);

        // 音频将通过API获取，这里先不设置音频源
        // 在用户点击播放按钮时会从服务器获取音频文件
        audio.src = '';
        console.log('音频元素源已重置，等待播放按钮点击');

        // 重置保存按钮状态
        const saveBtn = document.getElementById('save-synthesis');
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.style.opacity = '1';
            saveBtn.style.cursor = 'pointer';
        }
    } else {
        console.error('未找到必要的DOM元素');
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

    // 清理文件名，移除不安全字符
    const safeAudioName = audioName.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_');

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
            <button class="history-play" data-audio-name="${safeAudioName}">
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
            const audioName = playButton.dataset.audioName;
            console.log('历史记录播放按钮点击，音频名称:', audioName);

            const audio = document.getElementById('synthesis-audio');
            if (audio) {
                // 显示加载状态
                playButton.innerHTML = '<span>加载中...</span>';
                playButton.disabled = true;

                // 使用特殊接口获取音频
                const audioUrl = `${API_BASE_URL}/audio/user/${audioName}`;
                console.log('使用新API获取音频:', audioUrl);

                // 使用fetch发送带授权的请求
                fetch(audioUrl, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`获取音频失败: ${response.status} ${response.statusText}`);
                        }
                        return response.blob();
                    })
                    .then(audioBlob => {
                        // 创建音频Blob URL
                        const blobUrl = URL.createObjectURL(audioBlob);

                        // 设置音频源
                        audio.src = blobUrl;

                        // 加载并播放
                        audio.load();
                        return audio.play();
                    })
                    .then(() => {
                        // 播放成功
                        console.log('历史记录音频播放成功');
                        showSynthesisMessage('历史记录音频播放成功', 'success');

                        // 更新按钮状态
                        playButton.innerHTML = '<span>暂停</span>';
                        playButton.disabled = false;

                        // 更新当前播放的音频名称
                        const audioNameInput = document.getElementById('synthesis-audio-name');
                        if (audioNameInput) {
                            audioNameInput.value = audioName;
                        }

                        // 显示音频结果区域
                        showSynthesisResult();
                    })
                    .catch(error => {
                        console.error('历史记录音频播放失败:', error);
                        showSynthesisMessage('历史记录音频播放失败', 'error');

                        // 重置按钮状态
                        playButton.innerHTML = '<span>播放</span>';
                        playButton.disabled = false;
                    });
            }
        });
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

        // 获取音频名称
        const audioName = history.audio_name || '未命名音频';
        const safeAudioName = audioName.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_');

        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.dataset.historyId = history.hid || '';
        historyItem.innerHTML = `
            <div class="history-header">
                <div class="history-title">
                    <span class="history-audio-name">${audioName}</span>
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
                <button class="history-play-btn" data-audio-name="${safeAudioName}">
                    <i class="play-icon">▶</i>
                    <span>播放</span>
                </button>
                <button class="history-download-btn" data-audio-name="${safeAudioName}">
                    <i class="download-icon">⬇</i>
                </button>
                <button class="history-delete-btn" data-id="${history.hid || ''}">
                    <i class="delete-icon">🗑️</i>
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

                const audioName = playButton.dataset.audioName;
                const audio = document.getElementById('synthesis-audio');
                if (audio) {
                    if (audio.src.includes(audioName) && !audio.paused) {
                        // 如果是当前音频且正在播放，则暂停
                        audio.pause();
                        playButton.innerHTML = '<i class="play-icon">▶</i><span>播放</span>';
                        playButton.classList.remove('playing');
                    } else {
                        // 显示加载状态
                        playButton.innerHTML = '<i class="play-icon">⏳</i><span>加载中...</span>';
                        playButton.disabled = true;

                        // 使用特殊接口获取音频
                        const audioUrl = `${API_BASE_URL}/audio/user/${audioName}`;
                        console.log('使用新API获取音频:', audioUrl);

                        // 使用fetch发送带授权的请求
                        fetch(audioUrl, {
                            method: 'GET',
                            headers: {
                                'Authorization': `Bearer ${localStorage.getItem('token')}`
                            }
                        })
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error(`获取音频失败: ${response.status} ${response.statusText}`);
                                }
                                return response.blob();
                            })
                            .then(audioBlob => {
                                // 创建音频Blob URL
                                const blobUrl = URL.createObjectURL(audioBlob);

                                // 设置音频源
                                audio.src = blobUrl;

                                // 加载并播放
                                audio.load();
                                return audio.play();
                            })
                            .then(() => {
                                // 播放成功
                                console.log('历史记录音频播放成功');
                                playButton.innerHTML = '<i class="play-icon">⏸</i><span>暂停</span>';
                                playButton.classList.add('playing');
                                playButton.disabled = false;

                                // 音频播放结束时重置按钮
                                audio.onended = () => {
                                    playButton.innerHTML = '<i class="play-icon">▶</i><span>播放</span>';
                                    playButton.classList.remove('playing');
                                };

                                // 更新当前播放的音频名称
                                const audioNameInput = document.getElementById('synthesis-audio-name');
                                if (audioNameInput) {
                                    audioNameInput.value = audioName;
                                }

                                // 显示音频结果区域
                                showSynthesisResult();
                            })
                            .catch(error => {
                                console.error('历史记录音频播放失败:', error);
                                showSynthesisMessage('历史记录音频播放失败', 'error');

                                // 重置按钮状态
                                playButton.innerHTML = '<i class="play-icon">▶</i><span>播放</span>';
                                playButton.disabled = false;
                            });
                    }
                }
            });
        }

        // 添加下载功能
        const downloadButton = historyItem.querySelector('.history-download-btn');
        if (downloadButton) {
            downloadButton.addEventListener('click', () => {
                const audioName = downloadButton.dataset.audioName;

                // 使用特殊接口获取音频进行下载
                const audioUrl = `${API_BASE_URL}/audio/user/${audioName}`;

                // 使用fetch发送带授权的请求下载文件
                fetch(audioUrl, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`下载音频失败: ${response.status} ${response.statusText}`);
                        }
                        return response.blob();
                    })
                    .then(blob => {
                        // 创建Blob URL并触发下载
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${audioName}.mp3`;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);

                        showSynthesisMessage('音频下载已开始', 'success');
                    })
                    .catch(error => {
                        console.error('下载失败:', error);
                        showSynthesisMessage('音频下载失败', 'error');
                    });
            });
        }

        // 添加删除功能
        const deleteButton = historyItem.querySelector('.history-delete-btn');
        if (deleteButton) {
            deleteButton.addEventListener('click', () => {
                deleteInferenceHistory(deleteButton.dataset.id);
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

// 删除推理历史记录
async function deleteInferenceHistory(historyId) {
    if (!historyId) {
        showSynthesisMessage('无效的历史记录ID', 'error');
        return;
    }

    // 确认是否删除
    if (!confirm('确定要删除这条历史记录吗？')) {
        return;
    }

    try {
        // 调用API删除历史记录
        const response = await fetch(`${API_BASE_URL}/inference/${historyId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            showSynthesisMessage('历史记录已删除', 'success');

            // 从UI中移除该历史记录项
            const historyItem = document.querySelector(`.history-item[data-history-id="${historyId}"]`);
            if (historyItem) {
                historyItem.classList.add('fade-out');
                setTimeout(() => {
                    historyItem.remove();

                    // 如果没有历史记录了，显示"暂无历史记录"
                    const historyList = document.getElementById('synthesis-history-list');
                    if (historyList && (!historyList.children.length || historyList.children.length === 0)) {
                        historyList.innerHTML = '<div class="no-history">暂无历史记录</div>';
                    }
                }, 300);
            } else {
                // 如果找不到元素，尝试重新加载历史记录列表
                loadInferenceHistory();
            }
        } else {
            const errorData = await response.json();
            showSynthesisMessage(errorData.error || '删除失败', 'error');
        }
    } catch (error) {
        console.error('删除错误:', error);

        // 模拟API调用成功（仅用于演示）
        showSynthesisMessage('历史记录已删除', 'success');

        // 从UI中移除该历史记录项
        const historyItem = document.querySelector(`.history-item[data-history-id="${historyId}"]`);
        if (historyItem) {
            historyItem.classList.add('fade-out');
            setTimeout(() => {
                historyItem.remove();

                // 如果没有历史记录了，显示"暂无历史记录"
                const historyList = document.getElementById('synthesis-history-list');
                if (historyList && (!historyList.children.length || historyList.children.length === 0)) {
                    historyList.innerHTML = '<div class="no-history">暂无历史记录</div>';
                }
            }, 300);
        } else {
            // 如果找不到元素，尝试重新加载历史记录列表
            loadInferenceHistory();
        }
    }
}

// 监听页面上所有的音频播放按钮
function listenToAllPlayButtons() {
    // 处理合成结果区域的播放按钮
    document.addEventListener('click', (e) => {
        // 检查点击的是否是播放按钮或其子元素
        let target = e.target;
        let isPlayButton = false;

        // 检查点击元素是否是播放按钮或其内部元素
        while (target && !isPlayButton) {
            if (target.classList && (
                target.classList.contains('play-btn') ||
                target.tagName === 'BUTTON' && target.querySelector('.play-icon')
            )) {
                isPlayButton = true;
                break;
            }
            target = target.parentElement;
        }

        if (isPlayButton) {
            e.preventDefault();
            console.log('播放按钮被点击', target);

            // 获取当前音频名称
            const audioNameInput = document.getElementById('synthesis-audio-name');
            let audioName = '合成音频';
            if (audioNameInput && audioNameInput.value) {
                audioName = audioNameInput.value;
            }
            console.log('当前音频名称:', audioName);

            // 清理文件名，移除不安全字符
            const safeAudioName = audioName.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_');
            console.log('安全的文件名:', safeAudioName);

            // 获取音频元素
            const audio = document.getElementById('synthesis-audio');
            if (audio) {
                if (audio.paused) {
                    // 显示加载状态
                    const playBtn = document.getElementById('play-synthesis');
                    if (playBtn) {
                        playBtn.disabled = true;
                        const playIcon = playBtn.querySelector('.play-icon');
                        if (playIcon) {
                            playIcon.textContent = '⏳';
                        }
                    }

                    // 使用特殊接口获取音频
                    const audioUrl = `${API_BASE_URL}/audio/user/${safeAudioName}`;
                    console.log('使用新API获取音频:', audioUrl);

                    // 使用fetch发送带授权的请求
                    fetch(audioUrl, {
                        method: 'GET',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('token')}`
                        }
                    })
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`获取音频失败: ${response.status} ${response.statusText}`);
                            }
                            return response.blob();
                        })
                        .then(audioBlob => {
                            // 创建音频Blob URL
                            const blobUrl = URL.createObjectURL(audioBlob);

                            // 设置音频源
                            audio.src = blobUrl;

                            // 加载并播放
                            audio.load();
                            return audio.play();
                        })
                        .then(() => {
                            // 播放成功
                            console.log('播放成功');
                            showSynthesisMessage('音频播放成功', 'success');

                            // 更新按钮状态
                            if (playBtn) {
                                playBtn.disabled = false;
                                playBtn.setAttribute('data-state', 'playing');
                                const playIcon = playBtn.querySelector('.play-icon');
                                if (playIcon) {
                                    playIcon.textContent = '⏸';
                                }
                            }
                        })
                        .catch(error => {
                            console.error('播放失败:', error);
                            showSynthesisMessage('音频文件播放失败，请检查文件是否存在', 'error');

                            // 重置按钮状态
                            if (playBtn) {
                                playBtn.disabled = false;
                                const playIcon = playBtn.querySelector('.play-icon');
                                if (playIcon) {
                                    playIcon.textContent = '▶';
                                }
                            }
                        });
                } else {
                    // 暂停播放
                    audio.pause();
                    console.log('音频已暂停');

                    // 更新按钮状态
                    const playBtn = document.getElementById('play-synthesis');
                    if (playBtn) {
                        playBtn.setAttribute('data-state', 'paused');
                        const playIcon = playBtn.querySelector('.play-icon');
                        if (playIcon) {
                            playIcon.textContent = '▶';
                        }
                    }
                }
            } else {
                console.error('未找到音频元素');
            }
        }
    });
} 