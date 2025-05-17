package controller

import (
	"voice_clone_fbm/backend/model"
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

// Register 用户注册
func Register(c *gin.Context) {
	var req RegisterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "请求参数错误"})
		return
	}

	// 检查用户名是否已存在
	if _, err := model.GetByUsername(req.Username); err == nil {
		c.JSON(400, gin.H{"error": "用户名已存在"})
		return
	}

	// 创建新用户
	user := &model.User{
		Username: req.Username,
		Password: req.Password,
		Nickname: req.Nickname,
		Email:    req.Email,
	}

	if err := user.Create(); err != nil {
		c.JSON(500, gin.H{"error": "创建用户失败"})
		return
	}

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

	// 获取用户
	user, err := model.GetByUsername(req.Username)
	if err != nil {
		c.JSON(400, gin.H{"error": "用户名或密码错误"})
		return
	}

	// 验证密码
	if !user.ValidatePassword(req.Password) {
		c.JSON(400, gin.H{"error": "用户名或密码错误"})
		return
	}

	// 生成JWT token
	token, err := utils.GenerateToken(user.Username)
	if err != nil {
		c.JSON(500, gin.H{"error": "生成token失败"})
		return
	}

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
