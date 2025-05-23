package model

import (
	"gorm.io/gorm"
)

// InferenceHistory 推理历史记录表
type InferenceHistory struct {
	HID        string  `gorm:"type:char(32);index;not null" json:"hid"` // 历史记录ID
	UID        string  `gorm:"type:char(32);index;not null" json:"uid"` // 用户ID
	MID        string  `gorm:"type:char(32);index;not null" json:"mid"` // 模型ID
	InputText  string  `gorm:"type:text;not null" json:"input_text"`    // 输入文本
	OutputPath string  `gorm:"type:varchar(255)" json:"output_path"`    // 输出音频路径
	AudioName  string  `gorm:"type:varchar(255)" json:"audio_name"`     // 音频名称
	Status     int8    `gorm:"type:tinyint;default:1" json:"status"`    // 状态：1处理中 2已完成 3失败
	Duration   float32 `gorm:"type:float" json:"duration"`              // 音频时长(秒)
	gorm.Model         // 包含ID、创建时间、更新时间、删除时间
}

// TableName 指定表名
func (h *InferenceHistory) TableName() string {
	return "inference_histories"
}

// Create 创建记录
func (h *InferenceHistory) Create(db *gorm.DB) error {
	return db.Create(h).Error
}

// Update 更新记录
func (h *InferenceHistory) Update(db *gorm.DB) error {
	return db.Save(h).Error
}

// Delete 删除记录
func (h *InferenceHistory) Delete(db *gorm.DB) error {
	return db.Delete(h).Error
}

// GetByID 根据ID获取记录
func (h *InferenceHistory) GetByID(db *gorm.DB, id uint) error {
	return db.First(h, id).Error
}

// GetByHID 根据历史记录ID获取记录
func (h *InferenceHistory) GetByHID(db *gorm.DB, hid string) error {
	return db.Where("hid = ?", hid).First(h).Error
}

// GetByUID 获取用户的所有推理记录
func (h *InferenceHistory) GetByUID(db *gorm.DB, uid string) ([]InferenceHistory, error) {
	var histories []InferenceHistory
	err := db.Where("uid = ?", uid).Order("created_at DESC").Find(&histories).Error
	return histories, err
}

// GetByMID 获取模型的所有推理记录
func (h *InferenceHistory) GetByMID(db *gorm.DB, mid string) ([]InferenceHistory, error) {
	var histories []InferenceHistory
	err := db.Where("mid = ?", mid).Order("created_at DESC").Find(&histories).Error
	return histories, err
}

// UpdateStatus 更新推理状态
func (h *InferenceHistory) UpdateStatus(db *gorm.DB, status int8) error {
	return db.Model(h).Update("status", status).Error
}

// GetRecentHistories 获取最近的推理记录
func (h *InferenceHistory) GetRecentHistories(db *gorm.DB, limit int) ([]InferenceHistory, error) {
	var histories []InferenceHistory
	err := db.Order("created_at DESC").Limit(limit).Find(&histories).Error
	return histories, err
}

// HistoryDetail 推理历史详细信息
type HistoryDetail struct {
	InferenceHistory
	Username  string `json:"username"`   // 用户名
	ModelName string `json:"model_name"` // 模型名称
	AudioName string `json:"audio_name"` // 音频名称
}

// GetHistoryDetailByHID 获取历史记录详细信息
func (h *InferenceHistory) GetHistoryDetailByHID(db *gorm.DB, hid string) (*HistoryDetail, error) {
	var detail HistoryDetail
	err := db.Table("inference_histories as h").
		Select("h.*, u.username, a.name as audio_name, m.model_name as model_name").
		Joins("LEFT JOIN users u ON h.uid = u.uid").
		Joins("LEFT JOIN user_audio_models m ON h.mid = m.mid").
		Joins("LEFT JOIN audios a ON m.a_id = a.a_id").
		Where("h.hid = ? AND h.deleted_at IS NULL", hid).
		First(&detail).Error
	return &detail, err
}

// GetHistoryDetailsByUID 获取用户的所有推理记录详细信息
func (h *InferenceHistory) GetHistoryDetailsByUID(db *gorm.DB, uid string) ([]HistoryDetail, error) {
	var details []HistoryDetail
	err := db.Table("inference_histories as h").
		Select("h.*").
		Where("h.uid = ? AND h.deleted_at IS NULL", uid).
		Order("h.created_at DESC").
		Find(&details).Error
	return details, err
}
