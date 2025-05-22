package service

import (
	"fmt"
	"path/filepath"
	"time"

	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/utils"

	"github.com/google/uuid"
	log "github.com/sirupsen/logrus"
)

// ModelTrainReq 模型训练请求参数
type ModelTrainReq struct {
	AID       string `json:"aid"`       // 音频ID
	ModelName string `json:"modelName"` // 模型名称
	UID       string `json:"uid"`       // 用户ID
	Params    string `json:"params"`    // 训练参数(JSON格式)
}

// ModelTrainResp 模型训练响应
type ModelTrainResp struct {
	MID       string    `json:"mid"`       // 模型ID
	ModelName string    `json:"modelName"` // 模型名称
	State     int8      `json:"state"`     // 训练状态
	CreatedAt time.Time `json:"createdAt"` // 创建时间
}

// StartModelTraining 开始模型训练
func StartModelTraining(req *ModelTrainReq) (*ModelTrainResp, error) {
	// 验证音频是否存在
	var audio model.Audio
	if err := audio.GetByAID(req.AID); err != nil {
		log.Errorf("音频不存在: %v", err)
		return nil, fmt.Errorf("音频不存在: %v", err)
	}

	// 验证用户是否有权限访问该音频
	var userAudio model.UserAudio
	if err := model.DB.Where("uid = ? AND a_id = ?", req.UID, req.AID).First(&userAudio).Error; err != nil {
		log.Errorf("用户无权访问该音频: %v", err)
		return nil, fmt.Errorf("用户无权访问该音频")
	}

	// 生成模型ID
	mid := uuid.New().String()

	// 生成模型保存路径
	// 示例: data/{username}/models/{model_name}_{timestamp}
	username := ""
	var user model.User
	if err := model.DB.Where("uid = ?", req.UID).First(&user).Error; err != nil {
		log.Warnf("无法获取用户名，将使用UID: %v", err)
		username = req.UID
	} else {
		username = user.Username
	}

	// 确保模型目录存在
	modelDir := filepath.Join(utils.DataDir, username, "models")
	if err := utils.CreateDirIfNotExist(modelDir); err != nil {
		log.Errorf("创建模型目录失败: %v", err)
		return nil, fmt.Errorf("创建模型目录失败: %v", err)
	}

	modelPath := filepath.Join(modelDir, fmt.Sprintf("%s_%d", req.ModelName, time.Now().Unix()))

	// 开始数据库事务
	tx := model.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 创建模型记录
	voiceModel := &model.VoiceModel{
		MID:       mid,
		AID:       req.AID,
		UID:       req.UID,
		ModelName: req.ModelName,
		ModelPath: modelPath,
		State:     1, // 待训练状态
		Params:    req.Params,
	}

	if err := voiceModel.Create(tx); err != nil {
		tx.Rollback()
		log.Errorf("创建模型记录失败: %v", err)
		return nil, fmt.Errorf("创建模型记录失败: %v", err)
	}

	// 创建用户音频模型关系记录
	userAudioModel := &model.UserAudioModel{
		MID:       mid,
		UID:       req.UID,
		AID:       req.AID,
		ModelName: req.ModelName,
	}

	if err := userAudioModel.Create(tx); err != nil {
		tx.Rollback()
		log.Errorf("创建用户音频模型关系记录失败: %v", err)
		return nil, fmt.Errorf("创建用户音频模型关系记录失败: %v", err)
	}

	// 提交事务
	if err := tx.Commit().Error; err != nil {
		log.Errorf("提交事务失败: %v", err)
		return nil, fmt.Errorf("提交事务失败: %v", err)
	}

	// TODO: 实际的模型训练逻辑
	// 这里应该调用Python模型训练服务
	// 可以是异步任务，使用消息队列或者HTTP请求

	log.Infof("模型训练任务已创建: MID=%s, 音频ID=%s, 用户ID=%s, 模型名称=%s",
		voiceModel.MID, voiceModel.AID, voiceModel.UID, voiceModel.ModelName)

	// 返回响应
	return &ModelTrainResp{
		MID:       voiceModel.MID,
		ModelName: voiceModel.ModelName,
		State:     voiceModel.State,
		CreatedAt: time.Now(),
	}, nil
}

// GetUserModels 获取用户的所有模型
func GetUserModels(uid string) ([]model.VoiceModel, error) {
	// 首先获取用户的所有模型关联
	var userAudioModel model.UserAudioModel
	relationships, err := userAudioModel.GetByUID(model.DB, uid)
	if err != nil {
		log.Errorf("获取用户模型关系失败: %v", err)
		return nil, fmt.Errorf("获取用户模型关系失败: %v", err)
	}

	if len(relationships) == 0 {
		// 用户没有模型，返回空数组
		return []model.VoiceModel{}, nil
	}

	// 收集所有模型ID
	var mids []string
	for _, rel := range relationships {
		mids = append(mids, rel.MID)
	}

	// 根据模型ID查询模型详情
	var models []model.VoiceModel
	if err := model.DB.Where("m_id IN ?", mids).Find(&models).Error; err != nil {
		log.Errorf("查询模型详情失败: %v", err)
		return nil, fmt.Errorf("查询模型详情失败: %v", err)
	}

	return models, nil
}
