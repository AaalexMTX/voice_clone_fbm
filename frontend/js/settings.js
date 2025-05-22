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

// 初始化设置页面
document.addEventListener('DOMContentLoaded', function () {
    const saveButton = document.getElementById('save-settings');
    if (saveButton) {
        saveButton.addEventListener('click', saveSettings);
    }
}); 