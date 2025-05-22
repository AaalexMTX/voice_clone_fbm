package controller

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
	"voice_clone_fbm/backend/model"

	"github.com/gin-gonic/gin"
)

// UploadAudio 上传音频文件
func UploadAudio(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		log.Printf("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	file, err := c.FormFile("audio")
	if err != nil {
		log.Printf("文件上传失败: %v", err)
		c.JSON(400, gin.H{"error": "文件上传失败"})
		return
	}

	log.Printf("用户 %s 正在上传文件: %s", username, file.Filename)

	// 检查文件类型
	ext := strings.ToLower(filepath.Ext(file.Filename))
	if ext != ".wav" && ext != ".mp3" {
		log.Printf("用户 %s 尝试上传不支持的文件类型: %s", username, ext)
		c.JSON(400, gin.H{"error": "不支持的文件类型，仅支持WAV和MP3"})
		return
	}

	// 确保用户目录存在
	if err := model.EnsureUserDir(username); err != nil {
		log.Printf("创建用户目录失败: %v", err)
		c.JSON(500, gin.H{"error": "创建用户目录失败"})
		return
	}

	// 使用原始文件名
	filePath := model.GetUserVoicePath(username, file.Filename)
	log.Printf("保存文件到路径: %s", filePath)

	// 保存文件
	if err := c.SaveUploadedFile(file, filePath); err != nil {
		log.Printf("保存文件失败: %v", err)
		c.JSON(500, gin.H{"error": "文件保存失败"})
		return
	}

	// 创建音频记录
	audio := &model.Audio{
		Name:     file.Filename,
		FilePath: filePath,
		FileType: strings.TrimPrefix(ext, "."),
		Status:   model.AudioProcessPending,
	}
	if err := audio.Create(model.DB); err != nil {
		log.Printf("创建音频记录失败: %v", err)
		c.JSON(500, gin.H{"error": "创建音频记录失败"})
		return
	}

	// 创建用户音频关联
	userAudio := &model.UserAudio{
		UID:    username,
		AID:    audio.AID,
		Status: model.AudioTrainPending,
	}
	if err := userAudio.Create(model.DB); err != nil {
		log.Printf("创建用户音频关联失败: %v", err)
		c.JSON(500, gin.H{"error": "创建用户音频关联失败"})
		return
	}

	log.Printf("音频上传成功: ID=%d, 用户=%s, 文件名=%s", audio.ID, username, file.Filename)

	c.JSON(200, gin.H{
		"message": "音频上传成功",
		"audio": gin.H{
			"id":        audio.ID,
			"name":      audio.Name,
			"status":    audio.Status,
			"fileType":  audio.FileType,
			"filePath":  audio.FilePath,
			"createdAt": time.Now(),
		},
	})
}

// GetUserAudios 获取用户的音频列表
func GetUserAudios(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		log.Printf("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	log.Printf("获取用户 %s 的音频列表", username)

	audios, err := model.GetUserAudios(username)
	if err != nil {
		log.Printf("获取音频列表失败: %v", err)
		c.JSON(500, gin.H{"error": "获取音频列表失败"})
		return
	}

	log.Printf("成功获取用户 %s 的音频列表，共 %d 条记录", username, len(audios))
	c.JSON(200, gin.H{"audios": audios})
}

// DeleteAudio 删除音频
func DeleteAudio(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		log.Printf("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	audioID := c.Param("id")
	log.Printf("用户 %s 请求删除音频 ID: %s", username, audioID)

	// 获取音频记录
	var audio model.Audio
	if err := audio.GetByAID(audioID); err != nil {
		log.Printf("音频不存在: %v", err)
		c.JSON(404, gin.H{"error": "音频不存在"})
		return
	}

	// 检查用户权限
	var userAudio model.UserAudio
	if err := model.DB.Where("uid = ? AND aid = ?", username, audioID).First(&userAudio).Error; err != nil {
		log.Printf("用户无权访问该音频: %v", err)
		c.JSON(403, gin.H{"error": "无权访问该音频"})
		return
	}

	// 删除音频文件和记录
	if err := audio.Delete(); err != nil {
		log.Printf("删除音频失败: %v", err)
		c.JSON(500, gin.H{"error": fmt.Sprintf("删除音频失败: %v", err)})
		return
	}

	log.Printf("成功删除音频: ID=%s, 用户=%s, 文件名=%s", audioID, username, audio.Name)
	c.JSON(200, gin.H{"message": "音频删除成功"})
}

// UpdateAudioContent 更新音频内容
func UpdateAudioContent(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		log.Printf("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	audioID := c.Param("id")
	log.Printf("用户 %s 请求更新音频 ID: %s 的内容", username, audioID)

	var requestBody struct {
		Content string `json:"content" binding:"required"`
	}

	if err := c.ShouldBindJSON(&requestBody); err != nil {
		log.Printf("请求参数错误: %v", err)
		c.JSON(400, gin.H{"error": "无效的请求参数"})
		return
	}

	// 获取音频记录
	var audio model.Audio
	if err := model.DB.Where("id = ? AND username = ?", audioID, username).First(&audio).Error; err != nil {
		log.Printf("音频不存在或无权访问: %v", err)
		c.JSON(404, gin.H{"error": "音频不存在"})
		return
	}

	// 更新内容
	if err := audio.UpdateContent(requestBody.Content); err != nil {
		log.Printf("更新音频内容失败: %v", err)
		c.JSON(500, gin.H{"error": "更新内容失败"})
		return
	}

	log.Printf("成功更新音频内容: ID=%s, 用户=%s", audioID, username)
	c.JSON(200, gin.H{
		"message": "内容更新成功",
		"audio": gin.H{
			"id":      audio.ID,
			"content": requestBody.Content,
		},
	})
}

// StreamAudio 流式播放音频
func StreamAudio(c *gin.Context) {
	username := c.GetString("username")
	if username == "" {
		log.Printf("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	audioID := c.Param("id")
	log.Printf("用户 %s 请求播放音频 ID: %s", username, audioID)

	// 获取音频记录
	var audio model.Audio
	if err := model.DB.Where("id = ? AND username = ?", audioID, username).First(&audio).Error; err != nil {
		log.Printf("音频不存在或无权访问: %v", err)
		c.JSON(404, gin.H{"error": "音频不存在"})
		return
	}

	// 检查文件是否存在
	if _, err := os.Stat(audio.FilePath); os.IsNotExist(err) {
		log.Printf("音频文件不存在: %s", audio.FilePath)
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
