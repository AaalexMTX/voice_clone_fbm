package controller

import (
	"fmt"
	"os"
	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/service"
	"voice_clone_fbm/backend/utils"

	"github.com/gin-gonic/gin"
)

// UploadAudio 上传音频文件
func UploadAudio(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	file, err := c.FormFile("audio")
	if err != nil {
		utils.LogError("文件上传失败: %v", err)
		c.JSON(400, gin.H{"error": "文件上传失败"})
		return
	}

	// 调用服务层处理上传逻辑
	req := &service.AudioUploadReq{
		File:     file,
		Username: username,
	}

	resp, err := service.UploadAudio(req)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{
		"message": "音频上传成功",
		"audio": gin.H{
			"id":        resp.AudioID,
			"name":      resp.Name,
			"status":    resp.Status,
			"fileType":  resp.FileType,
			"filePath":  resp.FilePath,
			"createdAt": resp.CreatedAt,
		},
	})
}

// GetUserAudios 获取用户的音频列表
func GetUserAudios(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		utils.LogWarning("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	// 调用服务层获取音频列表
	audios, err := service.GetUserAudios(username)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// 转换为前端需要的格式，确保AID映射为id
	var response []gin.H
	for _, audio := range audios {
		response = append(response, gin.H{
			"id":        audio.AID, // 明确映射AID为id
			"name":      audio.Name,
			"fileType":  audio.FileType,
			"duration":  audio.Duration,
			"content":   audio.Content,
			"status":    audio.Status,
			"createdAt": audio.CreatedAt,
		})
	}

	c.JSON(200, gin.H{"audios": response})
}

// DeleteAudio 删除音频
func DeleteAudio(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		utils.LogWarning("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	audioID := c.Param("id")

	// 调用服务层删除音频
	if err := service.DeleteAudio(username, audioID); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{"message": "音频删除成功"})
}

// UpdateAudioContent 更新音频内容
func UpdateAudioContent(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		utils.LogWarning("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	audioID := c.Param("id")
	utils.LogInfo("用户 %s 请求更新音频 ID: %s 的内容", username, audioID)

	// 检查audioID是否有效
	if audioID == "" || audioID == "undefined" || audioID == "null" {
		utils.LogError("无效的音频ID: %s", audioID)
		c.JSON(400, gin.H{"error": "无效的音频ID"})
		return
	}

	var requestBody struct {
		Content string `json:"content" binding:"required"`
	}

	if err := c.ShouldBindJSON(&requestBody); err != nil {
		utils.LogError("请求参数错误: %v", err)
		c.JSON(400, gin.H{"error": "无效的请求参数"})
		return
	}

	// 调用服务层更新音频内容
	err := service.UpdateAudioContent(audioID, requestBody.Content, username)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{
		"message": "内容更新成功",
		"audio": gin.H{
			"id":      audioID,
			"content": requestBody.Content,
		},
	})
}

// StreamAudio 流式播放音频
func StreamAudio(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		utils.LogWarning("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	audioID := c.Param("id")
	utils.LogInfo("用户 %s 请求播放音频 ID: %s", username, audioID)

	// 检查audioID是否有效
	if audioID == "" || audioID == "undefined" || audioID == "null" {
		utils.LogError("无效的音频ID: %s", audioID)
		c.JSON(400, gin.H{"error": "无效的音频ID"})
		return
	}

	// 1. 根据用户名获取用户信息
	user, err := model.GetByUsername(username)
	if err != nil {
		utils.LogError("获取用户信息失败: %v", err)
		c.JSON(500, gin.H{"error": "获取用户信息失败"})
		return
	}

	// 2. 获取音频记录
	var audio model.Audio
	if err := audio.GetByAID(audioID); err != nil {
		utils.LogError("音频不存在: %v", err)
		c.JSON(404, gin.H{"error": "音频不存在"})
		return
	}

	// 3. 检查用户权限
	var userAudio model.UserAudio
	if err := model.DB.Where("uid = ? AND a_id = ?", user.UID, audio.AID).First(&userAudio).Error; err != nil {
		utils.LogWarning("用户无权访问该音频: %v", err)
		c.JSON(403, gin.H{"error": "无权访问该音频"})
		return
	}

	// 4. 检查文件是否存在
	if _, err := os.Stat(audio.FilePath); os.IsNotExist(err) {
		utils.LogError("音频文件不存在: %s", audio.FilePath)
		c.JSON(404, gin.H{"error": "音频文件不存在"})
		return
	}

	// 设置响应头
	c.Header("Content-Type", fmt.Sprintf("audio/%s", audio.FileType))
	c.Header("Accept-Ranges", "bytes")
	c.Header("Content-Disposition", fmt.Sprintf("inline; filename=%s", audio.Name))

	// 使用 c.File 提供文件下载
	c.File(audio.FilePath)
}
