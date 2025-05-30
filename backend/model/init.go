package model

import (
	"fmt"
	"voice_clone_fbm/backend/config"

	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

var DB *gorm.DB

// InitDB 初始化数据库连接
func InitDB() error {
	var err error
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=%s&parseTime=%v&loc=%s",
		config.GlobalConfig.Database.Username,
		config.GlobalConfig.Database.Password,
		config.GlobalConfig.Database.Host,
		config.GlobalConfig.Database.Port,
		config.GlobalConfig.Database.DBName,
		config.GlobalConfig.Database.Charset,
		config.GlobalConfig.Database.ParseTime,
		config.GlobalConfig.Database.Loc,
	)

	DB, err = gorm.Open(mysql.Open(dsn), &gorm.Config{
		DisableForeignKeyConstraintWhenMigrating: true, // 禁用外键约束
	})
	if err != nil {
		return fmt.Errorf("连接数据库失败: %v", err)
	}

	// 自动迁移
	if err := autoMigrate(); err != nil {
		return fmt.Errorf("数据库迁移失败: %v", err)
	}

	return nil
}

// autoMigrate 自动迁移数据库结构
func autoMigrate() error {
	// 获取当前数据库中已存在的表
	tables, err := DB.Migrator().GetTables()
	if err != nil {
		return err
	}

	// 用户表迁移
	if !contains(tables, "users") {
		if err := DB.Migrator().CreateTable(&User{}); err != nil {
			return err
		}
	} else {
		if err := DB.AutoMigrate(&User{}); err != nil {
			return err
		}
	}

	// 音频表迁移
	if !contains(tables, "audios") {
		if err := DB.Migrator().CreateTable(&Audio{}); err != nil {
			return err
		}
	} else {
		if err := DB.AutoMigrate(&Audio{}); err != nil {
			return err
		}
	}

	// 用户素材表迁移
	if !contains(tables, "user_audios") {
		if err := DB.Migrator().CreateTable(&UserAudio{}); err != nil {
			return err
		}
	} else {
		if err := DB.AutoMigrate(&UserAudio{}); err != nil {
			return err
		}
	}

	// 语音模型表迁移
	if !contains(tables, "models") {
		if err := DB.Migrator().CreateTable(&VoiceModel{}); err != nil {
			return err
		}
	} else {
		if err := DB.AutoMigrate(&VoiceModel{}); err != nil {
			return err
		}
	}

	// 用户音频模型关系表迁移
	if !contains(tables, "user_audio_models") {
		if err := DB.Migrator().CreateTable(&UserAudioModel{}); err != nil {
			return err
		}
	} else {
		if err := DB.AutoMigrate(&UserAudioModel{}); err != nil {
			return err
		}
	}

	return nil
}

// contains 检查切片中是否包含某个字符串
func contains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}
