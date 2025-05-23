package model

import (
	"gorm.io/gorm"
)

// VoiceModel 语音模型表
type VoiceModel struct {
	MID       string `gorm:"type:char(32);unique;index;not null" json:"mid"` // 模型唯一ID
	AID       string `gorm:"type:char(32);index;not null" json:"aid"`        // 音频唯一ID
	UID       string `gorm:"type:char(32);index;not null" json:"uid"`        // 用户唯一ID
	ModelName string `gorm:"type:varchar(50);not null" json:"model_name"`    // 模型名称
	ModelPath string `gorm:"type:varchar(255)" json:"model_path"`            // 模型文件路径
	State     int8   `gorm:"type:tinyint;default:1" json:"state"`            // 训练状态：1待训练 2已完成 3失败
	Params    string `gorm:"type:text" json:"params"`                        // 训练参数(JSON)
	ErrorMsg  string `gorm:"type:varchar(512)" json:"error_msg"`             // 错误信息
	gorm.Model
}

// TableName 设置表名
func (VoiceModel) TableName() string {
	return "models"
}

// Create 创建语音模型记录
func (vm *VoiceModel) Create(db *gorm.DB) error {
	return db.Create(vm).Error
}

// Update 更新语音模型记录
func (vm *VoiceModel) Update(db *gorm.DB) error {
	return db.Save(vm).Error
}

// Delete 删除语音模型记录(软删除)
func (vm *VoiceModel) Delete(db *gorm.DB) error {
	return db.Delete(vm).Error
}

// GetByMID 根据模型ID获取模型信息
func (vm *VoiceModel) GetByMID(mid string) error {
	return DB.Where("m_id = ?", mid).First(vm).Error
}

// GetUserModels 获取用户的所有语音模型
func GetUserModels(uid string) ([]VoiceModel, error) {
	var models []VoiceModel
	err := DB.Where("uid = ?", uid).Order("created_at DESC").Find(&models).Error
	return models, err
}

// GetModelsByAID 根据音频ID获取相关模型
func GetModelsByAID(aid string) ([]VoiceModel, error) {
	var models []VoiceModel
	err := DB.Where("a_id = ?", aid).Find(&models).Error
	return models, err
}

// InitVoiceModelTable 初始化语音模型表
func InitVoiceModelTable(db *gorm.DB) error {
	if !db.Migrator().HasTable(&VoiceModel{}) {
		if err := db.Migrator().CreateTable(&VoiceModel{}); err != nil {
			return err
		}
	}
	return db.AutoMigrate(&VoiceModel{})
}
