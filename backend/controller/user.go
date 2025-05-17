package controller

import (
	"voice_clone_fbm/backend/model"

	"github.com/gin-gonic/gin"
)

// UpdateUserInfo 更新用户信息
func UpdateUserInfo(c *gin.Context) {
	username := c.GetString("username") // 从JWT中获取用户名
	if username == "" {
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	var updateData struct {
		Nickname string `json:"nickname"`
		Email    string `json:"email"`
	}

	if err := c.ShouldBindJSON(&updateData); err != nil {
		c.JSON(400, gin.H{"error": "无效的请求数据"})
		return
	}

	user, err := model.GetByUsername(username)
	if err != nil {
		c.JSON(404, gin.H{"error": "用户不存在"})
		return
	}

	// 更新用户信息
	user.Nickname = updateData.Nickname
	user.Email = updateData.Email

	if err := user.Update(); err != nil {
		c.JSON(500, gin.H{"error": "更新用户信息失败"})
		return
	}

	c.JSON(200, gin.H{
		"message": "用户信息更新成功",
		"user": gin.H{
			"username": user.Username,
			"nickname": user.Nickname,
			"email":    user.Email,
		},
	})
}
