package controller

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"
	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/service"
	"voice_clone_fbm/backend/utils"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// StartTraining 开始模型训练
func StartTraining(c *gin.Context) {
	// 获取当前用户信息
	uid := c.GetString("uid")
	if uid == "" {
		utils.LogWarning("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	// 解析请求参数
	var req struct {
		AID       string `json:"aid" binding:"required"`       // 音频ID
		ModelName string `json:"modelName" binding:"required"` // 模型名称
		Params    string `json:"params"`                       // 训练参数(JSON格式)
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		utils.LogError("请求参数错误: %v", err)
		c.JSON(400, gin.H{"error": "无效的请求参数"})
		return
	}

	// 调用服务层开始训练
	trainReq := &service.ModelTrainReq{
		AID:       req.AID,
		ModelName: req.ModelName,
		UID:       uid,
		Params:    req.Params,
	}

	resp, err := service.StartModelTraining(trainReq)
	if err != nil {
		utils.LogError("开始模型训练失败: %v", err)
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// 返回响应
	c.JSON(200, gin.H{
		"message": "模型训练任务已创建",
		"model": gin.H{
			"mid":       resp.MID,
			"modelName": resp.ModelName,
			"state":     resp.State,
			"createdAt": resp.CreatedAt,
		},
	})
}

// GetUserModels 获取用户的所有模型
func GetUserModels(c *gin.Context) {
	uid := c.GetString("uid")
	if uid == "" {
		utils.LogWarning("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	// 调用服务层获取用户模型
	models, err := service.GetUserModels(uid)
	if err != nil {
		utils.LogError("获取用户模型失败: %v", err)
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// 获取用户音频模型关系
	var userAudioModel model.UserAudioModel
	relationships, err := userAudioModel.GetByUID(model.DB, uid)
	if err != nil {
		utils.LogError("获取用户音频模型关系失败: %v", err)
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// 创建MID到关系的映射
	midToRelationship := make(map[string]model.UserAudioModel)
	for _, rel := range relationships {
		midToRelationship[rel.MID] = rel
	}

	// 查找相关的音频信息
	type AudioInfo struct {
		AID     string `json:"aid"`
		Name    string `json:"name"`
		Content string `json:"content"`
	}
	aidToAudio := make(map[string]AudioInfo)

	// 收集所有需要查询的AID
	var aids []string
	for _, rel := range relationships {
		aids = append(aids, rel.AID)
	}

	// 批量查询音频信息
	if len(aids) > 0 {
		var audios []model.Audio
		if err := model.DB.Where("a_id IN ?", aids).Find(&audios).Error; err == nil {
			for _, audio := range audios {
				aidToAudio[audio.AID] = AudioInfo{
					AID:     audio.AID,
					Name:    audio.Name,
					Content: audio.Content,
				}
			}
		}
	}

	// 转换为前端需要的格式
	var response []gin.H
	for _, m := range models {
		modelResp := gin.H{
			"mid":       m.MID,
			"modelName": m.ModelName,
			"state":     m.State,
			"params":    m.Params,
			"errorMsg":  m.ErrorMsg,
			"createdAt": m.CreatedAt,
		}

		// 添加关联的音频信息
		if rel, exists := midToRelationship[m.MID]; exists {
			modelResp["aid"] = rel.AID

			// 添加音频详细信息
			if audio, exists := aidToAudio[rel.AID]; exists {
				modelResp["audioName"] = audio.Name
				modelResp["audioContent"] = audio.Content
			}
		}

		response = append(response, modelResp)
	}

	c.JSON(200, gin.H{"models": response})
}

// CreateTrainingModel 创建一个处于训练中状态的模型（直接设置state=2）
func CreateTrainingModel(c *gin.Context) {
	// 获取当前用户信息
	uid := c.GetString("uid")
	if uid == "" {
		utils.LogWarning("未授权的访问尝试")
		c.JSON(401, gin.H{"error": "未授权"})
		return
	}

	// 解析请求参数
	var req struct {
		AID       string `json:"aid" binding:"required"`       // 音频ID
		ModelName string `json:"modelName" binding:"required"` // 模型名称
		Params    string `json:"params"`                       // 训练参数(JSON格式)
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		utils.LogError("请求参数错误: %v", err)
		c.JSON(400, gin.H{"error": "无效的请求参数"})
		return
	}

	// 验证音频是否存在
	var audio model.Audio
	if err := audio.GetByAID(req.AID); err != nil {
		utils.LogError("音频不存在: %v", err)
		c.JSON(400, gin.H{"error": fmt.Sprintf("音频不存在: %v", err)})
		return
	}

	// 验证用户是否有权限访问该音频
	var userAudio model.UserAudio
	if err := model.DB.Where("uid = ? AND a_id = ?", uid, req.AID).First(&userAudio).Error; err != nil {
		utils.LogWarning("用户无权访问该音频: %v", err)
		c.JSON(403, gin.H{"error": "用户无权访问该音频"})
		return
	}

	// 生成模型ID - 使用uuid
	mid := strings.ReplaceAll(uuid.New().String(), "-", "")

	// 生成模型保存路径
	username := ""
	var user model.User
	if err := model.DB.Where("uid = ?", uid).First(&user).Error; err != nil {
		utils.LogWarning("无法获取用户名，将使用UID: %v", err)
		username = uid
	} else {
		username = user.Username
	}

	// 确保模型目录存在
	modelDir := filepath.Join(utils.DataDir, username, "models")
	if err := utils.CreateDirIfNotExist(modelDir); err != nil {
		utils.LogError("创建模型目录失败: %v", err)
		c.JSON(500, gin.H{"error": fmt.Sprintf("创建模型目录失败: %v", err)})
		return
	}

	modelPath := filepath.Join(modelDir, fmt.Sprintf("%s_%d", req.ModelName, time.Now().Unix()))

	// 开始数据库事务
	tx := model.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 创建模型记录 - 直接设置state=2（训练中）
	voiceModel := &model.VoiceModel{
		MID:       mid,
		AID:       req.AID,
		UID:       uid,
		ModelName: req.ModelName,
		ModelPath: modelPath,
		State:     1, // 直接设置为训练中状态
		Params:    req.Params,
	}

	if err := voiceModel.Create(tx); err != nil {
		tx.Rollback()
		utils.LogError("创建模型记录失败: %v", err)
		c.JSON(500, gin.H{"error": fmt.Sprintf("创建模型记录失败: %v", err)})
		return
	}

	// 创建用户音频模型关系记录
	userAudioModel := &model.UserAudioModel{
		MID:       mid,
		UID:       uid,
		AID:       req.AID,
		ModelName: req.ModelName,
	}

	if err := userAudioModel.Create(tx); err != nil {
		tx.Rollback()
		utils.LogError("创建用户音频模型关系记录失败: %v", err)
		c.JSON(500, gin.H{"error": fmt.Sprintf("创建用户音频模型关系记录失败: %v", err)})
		return
	}

	// 提交事务
	if err := tx.Commit().Error; err != nil {
		utils.LogError("提交事务失败: %v", err)
		c.JSON(500, gin.H{"error": fmt.Sprintf("提交事务失败: %v", err)})
		return
	}

	// 返回响应
	c.JSON(200, gin.H{
		"message": "训练中模型已创建",
		"model": gin.H{
			"mid":       voiceModel.MID,
			"modelName": voiceModel.ModelName,
			"state":     voiceModel.State,
			"createdAt": time.Now(),
		},
	})
}
