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

    // 调试信息：打印接收到的音频数据
    console.log('收到的音频列表数据:', audios);

    audios.forEach((audio, index) => {
        const audioElement = document.createElement('div');
        audioElement.className = 'voice-item';

        // 创建唯一ID用于音频元素和波形图
        const audioId = `audio-${audio.id}`;
        const waveformId = `waveform-${audio.id}`;

        audioElement.innerHTML = `
            <div class="voice-player">
                <div class="audio-name">🎧 ${audio.name || `音频${index + 1}`}</div>
                <div class="audio-player-container">
                    <div class="audio-controls">
                        <button class="play-btn" data-audio-id="${audioId}" data-audio-real-id="${audio.id}">
                            <i class="play-icon">▶</i>
                        </button>
                        <div class="audio-progress">
                            <div class="progress-bar">
                                <div class="progress-current" id="progress-${audioId}"></div>
                            </div>
                            <div class="time-display">
                                <span class="current-time" id="current-${audioId}">0:00</span>
                                <span class="total-time" id="total-${audioId}">0:00</span>
                            </div>
                        </div>
                    </div>
                    <div class="waveform-container" id="${waveformId}"></div>
                    <audio id="${audioId}" preload="metadata" crossorigin="anonymous"></audio>
                </div>
            </div>
            <div class="voice-content">
                <textarea
                    class="content-textarea"
                    placeholder="请输入需要校对的文本..."
                >${audio.content || ''}</textarea>
            </div>
            <div class="voice-actions">
                <button class="update-btn" onclick="updateAudioContent('${audio.id}', this.parentElement.previousElementSibling.querySelector('.content-textarea').value)">
                    ✓ 更新文本
                </button>
                <button class="delete-btn" onclick="deleteAudio('${audio.id}')">
                    ✕ 删除音频
                </button>
            </div>
        `;
        container.appendChild(audioElement);

        // 添加淡入动画延迟
        audioElement.style.animationDelay = `${index * 0.1}s`;

        // 初始化音频播放器
        initAudioPlayer(audioId, audio.id, waveformId);
    });
}

// 初始化音频播放器
function initAudioPlayer(audioId, realAudioId, waveformId) {
    const audio = document.getElementById(audioId);
    const playBtn = document.querySelector(`button[data-audio-id="${audioId}"]`);
    const progressBar = document.getElementById(`progress-${audioId}`);
    const currentTimeDisplay = document.getElementById(`current-${audioId}`);
    const totalTimeDisplay = document.getElementById(`total-${audioId}`);
    const waveformContainer = document.getElementById(waveformId);
    
    if (!audio || !playBtn || !progressBar || !currentTimeDisplay || !totalTimeDisplay) {
        console.error('音频播放器初始化失败：元素未找到', audioId);
        return;
    }

    // 记录音频是否已加载
    let audioLoaded = false;
    let audioBlob = null;

    // 播放/暂停按钮点击事件
    playBtn.addEventListener('click', async () => {
        console.log(`播放按钮点击: ${audioId}, 当前状态:`, audio.paused ? '已暂停' : '正在播放');
        
        // 如果音频尚未加载，先加载音频
        if (!audioLoaded) {
            try {
                playBtn.disabled = true;
                playBtn.querySelector('.play-icon').textContent = '⏳';
                
                // 使用fetch API获取音频数据，并添加认证头
                const response = await fetch(`${API_BASE_URL}/audio/stream/${realAudioId}`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`获取音频失败: ${response.status} ${response.statusText}`);
                }
                
                // 获取音频数据并创建Blob URL
                audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                
                // 设置音频源
                audio.src = audioUrl;
                audioLoaded = true;
                
                // 确保音频元素正确加载
                audio.load();
                
                console.log(`音频 ${audioId} 加载成功`);
            } catch (error) {
                console.error(`音频 ${audioId} 加载失败:`, error);
                alert('音频加载失败，请刷新页面重试');
                playBtn.disabled = false;
                playBtn.querySelector('.play-icon').textContent = '▶';
                return;
            }
        }
        
        if (audio.paused) {
            // 暂停所有其他音频
            document.querySelectorAll('audio').forEach(a => {
                if (a.id !== audioId && !a.paused) {
                    a.pause();
                    const otherBtn = document.querySelector(`button[data-audio-id="${a.id}"]`);
                    if (otherBtn) {
                        otherBtn.querySelector('.play-icon').textContent = '▶';
                    }
                }
            });

            // 播放当前音频
            const playPromise = audio.play();
            
            // 处理播放Promise
            if (playPromise !== undefined) {
                playPromise.then(() => {
                    // 播放成功
                    console.log(`音频 ${audioId} 开始播放`);
                    playBtn.querySelector('.play-icon').textContent = '⏸';
                }).catch(error => {
                    // 播放失败
                    console.error(`音频 ${audioId} 播放失败:`, error);
                    alert('音频播放失败，请刷新页面重试');
                });
            }
        } else {
            // 暂停当前音频
            audio.pause();
            playBtn.querySelector('.play-icon').textContent = '▶';
        }
    });

    // 添加错误处理
    audio.addEventListener('error', (e) => {
        console.error('音频加载错误:', e);
        totalTimeDisplay.textContent = '加载失败';
        playBtn.disabled = false;
        playBtn.querySelector('.play-icon').textContent = '▶';
    });

    // 音频加载元数据后设置总时长
    audio.addEventListener('loadedmetadata', () => {
        console.log(`音频 ${audioId} 元数据加载完成，时长:`, audio.duration);
        playBtn.disabled = false;
        playBtn.querySelector('.play-icon').textContent = '▶';
        
        // 检查时长是否为NaN或0，这是常见的元数据加载失败情况
        if (isNaN(audio.duration) || audio.duration === 0) {
            console.warn(`音频 ${audioId} 时长无效:`, audio.duration);
            return;
        }
        
        totalTimeDisplay.textContent = formatTime(audio.duration);

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
        progressBar.style.width = `${progress}%`;
        currentTimeDisplay.textContent = formatTime(audio.currentTime);
    });

    // 音频播放结束时重置
    audio.addEventListener('ended', () => {
        playBtn.querySelector('.play-icon').textContent = '▶';
        progressBar.style.width = '0%';
        audio.currentTime = 0;
    });

    // 点击进度条跳转
    const progressContainer = progressBar.parentElement;
    progressContainer.addEventListener('click', (e) => {
        // 检查音频是否已加载
        if (!audioLoaded || isNaN(audio.duration) || audio.duration === 0) {
            console.warn('无法跳转：音频尚未加载或时长无效');
            return;
        }
        
        const rect = progressContainer.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        audio.currentTime = pos * audio.duration;
    });
    
    // 添加可以播放事件监听
    audio.addEventListener('canplay', () => {
        console.log(`音频 ${audioId} 可以播放了，时长:`, audio.duration);
        if (!isNaN(audio.duration) && audio.duration > 0) {
            totalTimeDisplay.textContent = formatTime(audio.duration);
        }
    });
}

// 创建简单的波形可视化
function createSimpleWaveform(container) {
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
    polyline.setAttribute('fill', 'rgba(0, 123, 255, 0.2)');
    polyline.setAttribute('stroke', 'rgba(0, 123, 255, 0.6)');
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

// 删除音频
async function deleteAudio(id) {
    // 验证id是否有效
    if (!id || id === 'undefined' || id === 'null') {
        alert('无效的音频ID');
        return;
    }

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

// 更新音频内容
async function updateAudioContent(audioId, content) {
    try {
        // 验证audioId是否有效
        if (!audioId || audioId === 'undefined' || audioId === 'null') {
            showMessage('无效的音频ID', 'error');
            return;
        }

        console.log('发送更新请求:', audioId, content);
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

// 文件上传区域的拖放功能
document.addEventListener('DOMContentLoaded', function () {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.getElementById('voice-file');

    if (!uploadArea || !fileInput) return; // 确保元素存在

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
        const fileInput = document.getElementById('voice-file');
        if (fileInput) fileInput.value = '';
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
    if (!uploadArea) return;

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
    if (!uploadArea) return;

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
    const fileInput = document.getElementById('voice-file');
    if (fileInput) fileInput.value = '';
    clearUploadProgress();
    showMessage('已取消上传', 'info');
} 