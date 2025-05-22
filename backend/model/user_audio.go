package model

import (
	"gorm.io/gorm"
)

// UserAudio 用户音频关系模型
type UserAudio struct {
	UID        string `gorm:"type:char(32);not null;index"` // 用户ID
	AID        string `gorm:"type:char(32);not null;index"` // 音频ID
	Name       string `gorm:"type:varchar(255)"`           // 音频名称
	gorm.Model        // ID, CreatedAt, UpdatedAt, DeletedAt
}

// Create 创建用户音频关系
func (ua *UserAudio) Create(db *gorm.DB) error {
	return db.Create(ua).Error
}

// ListUserAudios 获取用户的所有音频（包含音频信息）
func ListUserAudios(uid string) ([]UserAudio, error) {
	var userAudios []UserAudio
	err := DB.Preload("Audio").Where("uid = ?", uid).Find(&userAudios).Error
	return userAudios, err
}

// ListUserAudiosWithInfo 获取用户的所有音频（包含用户和音频信息）
func ListUserAudiosWithInfo(uid string) ([]UserAudio, error) {
	var userAudios []UserAudio
	err := DB.Preload("User").Preload("Audio").Where("uid = ?", uid).Find(&userAudios).Error
	return userAudios, err
}

// GetUserAudioRelationByID 根据ID获取用户音频
func GetUserAudioRelationByID(id uint) (*UserAudio, error) {
	var userAudio UserAudio
	err := DB.Preload("User").Preload("Audio").First(&userAudio, id).Error
	return &userAudio, err
}

// GetUserAudioRelation 根据音频ID获取用户音频
func GetUserAudioRelation(uid string, aid string) (*UserAudio, error) {
	var userAudio UserAudio
	err := DB.Preload("User").Preload("Audio").Where("uid = ? AND a_id = ?", uid, aid).First(&userAudio).Error
	return &userAudio, err
}

// Update 更新用户音频信息
func (ua *UserAudio) Update() error {
	return DB.Save(ua).Error
}

// Delete 删除用户音频关系
func (ua *UserAudio) Delete() error {
	return DB.Delete(ua).Error
}


// BatchCreateUserAudioRelations 批量创建用户音频关系
func BatchCreateUserAudioRelations(userAudios []*UserAudio) error {
	return DB.Create(&userAudios).Error
}

// BatchDeleteUserAudioRelations 批量删除用户音频关系
func BatchDeleteUserAudioRelations(uid string, aids []string) error {
	return DB.Where("uid = ? AND a_id IN ?", uid, aids).Delete(&UserAudio{}).Error
}

// CountUserAudioRelations 统计用户音频数量
func CountUserAudioRelations(uid string) (int64, error) {
	var count int64
	err := DB.Model(&UserAudio{}).Where("uid = ?", uid).Count(&count).Error
	return count, err
}

// ListUserAudiosByStatus 获取指定状态的用户音频（包含音频信息）
func ListUserAudiosByStatus(uid string, status int8) ([]UserAudio, error) {
	var userAudios []UserAudio
	err := DB.Preload("Audio").Where("uid = ? AND status = ?", uid, status).Find(&userAudios).Error
	return userAudios, err
}

// ListUserAudiosByStatusWithInfo 获取指定状态的用户音频（包含用户和音频信息）
func ListUserAudiosByStatusWithInfo(uid string, status int8) ([]UserAudio, error) {
	var userAudios []UserAudio
	err := DB.Preload("User").Preload("Audio").Where("uid = ? AND status = ?", uid, status).Find(&userAudios).Error
	return userAudios, err
}
