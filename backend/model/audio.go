package model

import (
	"fmt"
	"os"
	"time"

	"gorm.io/gorm"
)

// Audio 音频素材模型
type Audio struct {
	Username    string     `gorm:"type:varchar(50);not null;index" json:"username"`  // 所属用户
	Name        string     `gorm:"type:varchar(255);not null" json:"name"`           // 音频名称
	FilePath    string     `gorm:"type:varchar(500);not null" json:"file_path"`      // 文件存储路径
	FileType    string     `gorm:"type:varchar(20);not null" json:"file_type"`       // 文件类型(wav/mp3)
	Duration    float64    `gorm:"type:decimal(10,2)" json:"duration"`               // 音频时长(秒)
	Status      string     `gorm:"type:varchar(20);default:'pending'" json:"status"` // 处理状态(pending/processing/completed/failed)
	ProcessedAt *time.Time `json:"processed_at"`                                     // 处理完成时间，使用指针允许null值
	Content     string     `json:"content"`                                          // 新增：音频对应的文本内容
	gorm.Model
}

// CreateAudio 创建音频记录
func CreateAudio(audio *Audio) error {
	return DB.Create(audio).Error
}

// GetUserAudios 获取用户的所有音频
func GetUserAudios(username string) ([]Audio, error) {
	var audios []Audio
	err := DB.Where("username = ?", username).Find(&audios).Error
	return audios, err
}

// UpdateAudioStatus 更新音频状态
func (a *Audio) UpdateStatus(status string) error {
	a.Status = status
	if status == "completed" {
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
