// API基础URL
const API_BASE_URL = 'http://localhost:8083/api';

// 显示登录框
function showLogin() {
    document.querySelector('.login-box').style.display = 'block';
    document.querySelector('.register-box').style.display = 'none';
}

// 显示注册框
function showRegister() {
    document.querySelector('.login-box').style.display = 'none';
    document.querySelector('.register-box').style.display = 'block';
}

// 登录函数
async function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        const response = await fetch(`${API_BASE_URL}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();

        if (response.ok) {
            // 登录成功，保存token和用户信息
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));
            // 跳转到主页
            window.location.href = '/dashboard.html';
        } else {
            alert(data.error || '登录失败');
        }
    } catch (error) {
        alert('网络错误，请稍后重试');
    }
}

// 注册函数
async function register() {
    const username = document.getElementById('reg-username').value;
    const password = document.getElementById('reg-password').value;
    const email = document.getElementById('reg-email').value;
    const nickname = document.getElementById('reg-nickname').value;

    try {
        const response = await fetch(`${API_BASE_URL}/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password, email, nickname }),
        });

        const data = await response.json();

        if (response.ok) {
            alert('注册成功，请登录');
            showLogin();
        } else {
            alert(data.error || '注册失败');
        }
    } catch (error) {
        alert('网络错误，请稍后重试');
    }
}

// 检查登录状态
function checkAuth() {
    const user = localStorage.getItem('user');
    if (user && window.location.pathname === '/index.html') {
        window.location.href = '/dashboard.html';
    }
}

// 页面加载时检查登录状态
checkAuth(); 