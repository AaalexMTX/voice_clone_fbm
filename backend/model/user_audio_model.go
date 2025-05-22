package model

import (
	"gorm.io/gorm"
)

// UserAudioModel 用户音频模型表
type UserAudioModel struct {
	MID       string         `gorm:"type:char(32);index;not null" json:"mid"`       // 模型ID
	UID       string         `gorm:"type:char(32);index;not null" json:"uid"`       // 用户ID
	AID       string         `gorm:"type:char(32);index;not null" json:"aid"`       // 音频ID
	ModelPath string         `gorm:"type:varchar(255)" json:"model_path"`           // 模型文件路径
	Status    int8           `gorm:"type:tinyint;default:1" json:"status"`          // 状态：1待训练 2已完成 3失败
	Params    string         `gorm:"type:text" json:"params"`                       // 训练参数(JSON)
	ErrorMsg  string         `gorm:"type:varchar(512)" json:"error_msg"`            // 错误信息
	gorm.Model
}

// TableName 指定表名
func (m *UserAudioModel) TableName() string {
	return "user_audio_models"
}

// Create 创建记录
func (m *UserAudioModel) Create(db *gorm.DB) error {
	return db.Create(m).Error
}

// Update 更新记录
func (m *UserAudioModel) Update(db *gorm.DB) error {
	return db.Save(m).Error
}

// Delete 删除记录
func (m *UserAudioModel) Delete(db *gorm.DB) error {
	return db.Delete(m).Error
}

// GetByID 根据ID获取记录
func (m *UserAudioModel) GetByID(db *gorm.DB, id uint) error {
	return db.First(m, id).Error
}

// GetByUID 获取用户的所有模型
func (m *UserAudioModel) GetByUID(db *gorm.DB, uid string) ([]UserAudioModel, error) {
	var models []UserAudioModel
	err := db.Where("uid = ?", uid).Find(&models).Error
	return models, err
}

// GetByAID 获取音频相关的所有模型
func (m *UserAudioModel) GetByAID(db *gorm.DB, aid string) ([]UserAudioModel, error) {
	var models []UserAudioModel
	err := db.Where("aid = ?", aid).Find(&models).Error
	return models, err
}

// UpdateStatus 更新模型状态
func (m *UserAudioModel) UpdateStatus(db *gorm.DB, status int8, errorMsg string) error {
	updates := map[string]interface{}{
		"status": status,
	}
	if errorMsg != "" {
		updates["error_msg"] = errorMsg
	}
	return db.Model(m).Updates(updates).Error
}

// UpdateProgress 更新训练进度
func (m *UserAudioModel) UpdateProgress(db *gorm.DB, progress float32) error {
	return db.Model(m).Update("progress", progress).Error
}