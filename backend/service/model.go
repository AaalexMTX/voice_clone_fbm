package service

import (
	"fmt"
	"path/filepath"
	"strings"
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

	// 生成模型ID - 使用与AID和UID相同的格式，移除UUID中的连字符
	mid := strings.ReplaceAll(uuid.New().String(), "-", "")

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

	// 获取音频文件路径
	audioPath := audio.FilePath
	if audioPath == "" {
		log.Errorf("音频文件路径为空")
		return nil, fmt.Errorf("音频文件路径为空")
	}

	// 调用模型层API进行训练
	// 创建模型客户端
	modelClient := NewModelClient()

	// 异步处理模型训练
	go func() {
		// 更新模型状态为训练中
		updateModelState(mid, 2, "") // 2表示训练中

		var embeddingID string
		var outputID string
		var err error

		// 1. 提取说话人嵌入
		log.Infof("开始提取说话人嵌入，音频路径: %s", audioPath)
		embeddingID, err = modelClient.ExtractEmbedding(audioPath)
		if err != nil {
			log.Errorf("提取说话人嵌入失败: %v", err)
			updateModelState(mid, 4, fmt.Sprintf("提取说话人嵌入失败: %v", err)) // 4表示失败
			return
		}

		// 2. 使用说话人嵌入合成语音（测试用）
		log.Infof("开始合成测试音频，文本: %s，说话人嵌入ID: %s", audio.Content, embeddingID)
		outputID, err = modelClient.SynthesizeSpeech(audio.Content, embeddingID)
		if err != nil {
			log.Errorf("合成测试音频失败: %v", err)
			updateModelState(mid, 4, fmt.Sprintf("合成测试音频失败: %v", err)) // 4表示失败
			return
		}

		// 3. 下载生成的音频
		outputPath := filepath.Join(modelDir, fmt.Sprintf("%s_sample.wav", req.ModelName))
		log.Infof("下载生成的音频到: %s", outputPath)
		err = modelClient.DownloadAudio(outputID, outputPath)
		if err != nil {
			log.Errorf("下载生成的音频失败: %v", err)
			updateModelState(mid, 4, fmt.Sprintf("下载生成的音频失败: %v", err)) // 4表示失败
			return
		}

		// 更新模型状态为完成
		updateModelState(mid, 3, "") // 3表示完成
		log.Infof("模型训练完成，模型ID: %s", mid)
	}()

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

// 更新模型状态
func updateModelState(mid string, state int8, errorMsg string) {
	var voiceModel model.VoiceModel
	if err := model.DB.Where("m_id = ?", mid).First(&voiceModel).Error; err != nil {
		log.Errorf("查询模型失败: %v", err)
		return
	}

	// 更新状态
	voiceModel.State = state
	voiceModel.ErrorMsg = errorMsg
	voiceModel.UpdatedAt = time.Now()

	if err := model.DB.Save(&voiceModel).Error; err != nil {
		log.Errorf("更新模型状态失败: %v", err)
	}
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
