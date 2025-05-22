package controller

import (
	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/service"
	"voice_clone_fbm/backend/utils"

	"github.com/gin-gonic/gin"
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
