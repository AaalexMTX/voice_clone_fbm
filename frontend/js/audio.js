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
    });
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

// 取消上传
function cancelUpload() {
    // 清除文件输入
    const fileInput = document.getElementById('voice-file');
    if (fileInput) fileInput.value = '';
    clearUploadProgress();
    showMessage('已取消上传', 'info');
} 