package controller

import (
	"voice_clone_fbm/backend/service"
	"voice_clone_fbm/backend/utils"

	"github.com/gin-gonic/gin"
)

type RegisterRequest struct {
	Username string `json:"username" binding:"required,min=3,max=50"`
	Password string `json:"password" binding:"required,min=6"`
	Nickname string `json:"nickname"`
	Email    string `json:"email" binding:"email"`
}

type LoginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type UpdateUserRequest struct {
	Nickname string `json:"nickname"`
	Email    string `json:"email"`
}

// UpdateUserInfo 更新用户信息
func UpdateUserInfo(c *gin.Context) {
	username := c.GetString("username") // 从JWT中获取用户名
	if username == "" {
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	var req UpdateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "无效的请求数据"})
		return
	}

	// 调用service层的更新服务
	updateReq := &service.UserUpdateReq{
		Nickname: req.Nickname,
		Email:    req.Email,
	}

	updatedUser, err := service.UpdateUser(username, updateReq)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	utils.LogInfo("用户信息更新成功: user: %v", updatedUser)
	c.JSON(200, gin.H{
		"message": "用户信息更新成功",
		"user": gin.H{
			"username": updatedUser.Username,
			"nickname": updatedUser.Nickname,
			"email":    updatedUser.Email,
		},
	})
}

// Register 用户注册
func Register(c *gin.Context) {
	var req RegisterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "请求参数错误"})
		return
	}

	// 调用service层的注册服务
	userReq := &service.UserRegisterReq{
		Username: req.Username,
		Password: req.Password,
		Nickname: req.Nickname,
		Email:    req.Email,
	}

	user, err := service.Register(userReq)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	utils.LogInfo("注册成功: user: %v", user)
	c.JSON(200, gin.H{
		"message": "注册成功",
		"user": gin.H{
			"id":       user.ID,
			"username": user.Username,
			"nickname": user.Nickname,
			"email":    user.Email,
		},
	})
}

// Login 用户登录
func Login(c *gin.Context) {
	var req LoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "请求参数错误"})
		return
	}

	// 调用service层的登录服务
	userReq := &service.UserLoginReq{
		Username: req.Username,
		Password: req.Password,
	}

	user, token, err := service.Login(userReq)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	utils.LogInfo("登录成功: user: %v, token: %v", user, token)
	c.JSON(200, gin.H{
		"message": "登录成功",
		"token":   token,
		"user": gin.H{
			"id":       user.ID,
			"username": user.Username,
			"nickname": user.Nickname,
			"email":    user.Email,
			"role":     user.Role,
		},
	})
}
