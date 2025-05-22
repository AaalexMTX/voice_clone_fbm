package model

import (
	"fmt"
	"os"
	"time"

	"strings"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// AudioProcessStatus 音频处理状态
const (
	AudioProcessPending  int8 = 1 // 待处理
	AudioProcessComplete int8 = 2 // 已完成
	AudioProcessFailed   int8 = 3 // 失败
)

// Audio 音频模型
type Audio struct {
	AID         string     `gorm:"type:char(32);not null;unique;index" json:"aid"` // 音频唯一标识
	Name        string     `gorm:"type:varchar(255);not null" json:"name"`         // 音频名称
	FilePath    string     `gorm:"type:varchar(255);not null" json:"file_path"`    // 文件存储路径
	FileType    string     `gorm:"type:varchar(20);not null" json:"file_type"`     // 文件类型(wav/mp3)
	Duration    float64    `gorm:"type:decimal(10,2)" json:"duration"`             // 音频时长(秒)
	Status      int8       `gorm:"type:tinyint;default:1" json:"status"`           // 处理状态(1:待处理 2:已完成 3:失败)
	ProcessedAt *time.Time `json:"processed_at"`                                   // 处理完成时间
	Content     string     `gorm:"type:text" json:"content"`                       // 音频对应的文本内容
	gorm.Model             // ID, CreatedAt, UpdatedAt, DeletedAt
}

// BeforeCreate 创建音频前生成AID
func (a *Audio) BeforeCreate(tx *gorm.DB) error {
	// 生成32位的UUID作为AID
	a.AID = strings.ReplaceAll(uuid.New().String(), "-", "")
	return nil
}

// Create 创建音频记录
func (a *Audio) Create(db *gorm.DB) error {
	return db.Create(a).Error
}

// GetAudioByAID 通过AID获取音频
func GetAudioByAID(aid string) (*Audio, error) {
	var audio Audio
	err := DB.Where("aid = ?", aid).First(&audio).Error
	return &audio, err
}

// Update 更新音频信息
func (a *Audio) Update() error {
	return DB.Save(a).Error
}

// Delete 删除音频
func (a *Audio) Delete() error {
	return DB.Delete(a).Error
}

// BatchDeleteAudios 批量删除音频
func BatchDeleteAudios(aids []string) error {
	return DB.Where("aid IN ?", aids).Delete(&Audio{}).Error
}

// UpdateAudioStatus 更新音频状态
func (a *Audio) UpdateStatus(status int8) error {
	a.Status = status
	if status == AudioProcessComplete {
		now := time.Now()
		a.ProcessedAt = &now
	}
	return DB.Save(a).Error
}

// DeleteAudio 删除音频记录和文件
func DeleteAudio(audio *Audio) error {
	// 删除物理文件
	if err := os.Remove(audio.FilePath); err != nil {
		return fmt.Errorf("删除文件失败: %v", err)
	}

	// 删除数据库记录
	return DB.Delete(audio).Error
}

// UpdateContent 更新音频内容
func (a *Audio) UpdateContent(content string) error {
	return DB.Model(a).Update("content", content).Error
}

// GetAudioByID 根据ID获取音频
func GetAudioByID(id uint) (*Audio, error) {
	var audio Audio
	if err := DB.First(&audio, id).Error; err != nil {
		return nil, err
	}
	return &audio, nil
}

// GetAudiosByStatus 获取指定状态的音频
func GetAudiosByStatus(status int8) ([]Audio, error) {
	var audios []Audio
	err := DB.Where("status = ?", status).Find(&audios).Error
	return audios, err
}

// IsPending 检查是否为待处理状态
func (a *Audio) IsPending() bool {
	return a.Status == AudioProcessPending
}

// IsComplete 检查是否为已完成状态
func (a *Audio) IsComplete() bool {
	return a.Status == AudioProcessComplete
}

// IsFailed 检查是否为失败状态
func (a *Audio) IsFailed() bool {
	return a.Status == AudioProcessFailed
}

// 无db参数的获取用户音频列表
func GetUserAudios(uid string) ([]Audio, error) {
	var audios []Audio
	err := DB.Table("audios as a").
		Joins("LEFT JOIN user_audios ua ON a.aid = ua.aid").
		Where("ua.uid = ? AND a.deleted_at IS NULL", uid).
		Order("a.created_at DESC").
		Find(&audios).Error
	return audios, err
}

// 无db参数的根据AID获取音频
func (a *Audio) GetByAID(aid string) error {
	return DB.Where("aid = ?", aid).First(a).Error
}
