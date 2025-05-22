package model

import (
	"gorm.io/gorm"
)

// AudioTrainStatus 音频训练状态
const (
	AudioTrainPending  int8 = 1 // 待训练
	AudioTrainTraining int8 = 2 // 训练中
	AudioTrainComplete int8 = 3 // 已完成
	AudioTrainFailed   int8 = 4 // 失败
)

// UserAudio 用户音频关系模型
type UserAudio struct {
	UID         string  `gorm:"type:char(32);not null;index" json:"uid"` // 用户ID
	AID         string  `gorm:"type:char(32);not null;index" json:"aid"` // 音频ID
	Name        string  `gorm:"type:varchar(255)" json:"name"`           // 音频名称
	Duration    float64 `gorm:"type:float" json:"duration"`              // 音频时长(秒)
	Status      int8    `gorm:"type:tinyint;default:1" json:"status"`    // 训练状态(1:待训练 2:训练中 3:已完成 4:失败)
	TrainParams string  `gorm:"type:text" json:"train_params,omitempty"` // 训练参数(JSON)
	ErrorMsg    string  `gorm:"type:text" json:"error_msg,omitempty"`    // 错误信息
	gorm.Model          // ID, CreatedAt, UpdatedAt, DeletedAt
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
	err := DB.Preload("User").Preload("Audio").Where("uid = ? AND aid = ?", uid, aid).First(&userAudio).Error
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

// UpdateStatus 更新音频状态
func (ua *UserAudio) UpdateStatus(status int8, errorMsg string) error {
	ua.Status = status
	if errorMsg != "" {
		ua.ErrorMsg = errorMsg
	}
	return ua.Update()
}

// BatchCreateUserAudioRelations 批量创建用户音频关系
func BatchCreateUserAudioRelations(userAudios []*UserAudio) error {
	return DB.Create(&userAudios).Error
}

// BatchDeleteUserAudioRelations 批量删除用户音频关系
func BatchDeleteUserAudioRelations(uid string, aids []string) error {
	return DB.Where("uid = ? AND aid IN ?", uid, aids).Delete(&UserAudio{}).Error
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

// IsPending 检查是否为待训练状态
func (ua *UserAudio) IsPending() bool {
	return ua.Status == AudioTrainPending
}

// IsTraining 检查是否为训练中状态
func (ua *UserAudio) IsTraining() bool {
	return ua.Status == AudioTrainTraining
}

// IsComplete 检查是否为已完成状态
func (ua *UserAudio) IsComplete() bool {
	return ua.Status == AudioTrainComplete
}

// IsFailed 检查是否为失败状态
func (ua *UserAudio) IsFailed() bool {
	return ua.Status == AudioTrainFailed
}

// ListPendingUserAudios 获取待训练的用户音频
func ListPendingUserAudios(uid string) ([]UserAudio, error) {
	return ListUserAudiosByStatus(uid, AudioTrainPending)
}

// ListTrainingUserAudios 获取训练中的用户音频
func ListTrainingUserAudios(uid string) ([]UserAudio, error) {
	return ListUserAudiosByStatus(uid, AudioTrainTraining)
}

// ListCompletedUserAudios 获取已完成的用户音频
func ListCompletedUserAudios(uid string) ([]UserAudio, error) {
	return ListUserAudiosByStatus(uid, AudioTrainComplete)
}

// ListFailedUserAudios 获取失败的用户音频
func ListFailedUserAudios(uid string) ([]UserAudio, error) {
	return ListUserAudiosByStatus(uid, AudioTrainFailed)
}
