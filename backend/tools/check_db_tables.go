package main

import (
	"fmt"
	"voice_clone_fbm/backend/config"
	"voice_clone_fbm/backend/model"
)

func main() {
	// 加载配置
	if err := config.LoadConfig("../config/config.yaml"); err != nil {
		fmt.Printf("加载配置失败: %v\n", err)
		return
	}

	// 初始化数据库连接
	if err := model.InitDB(); err != nil {
		fmt.Printf("初始化数据库失败: %v\n", err)
		return
	}

	db := model.GetDB()

	// 检查inference_histories表结构
	type Column struct {
		Field   string
		Type    string
		Null    string
		Key     string
		Default interface{}
		Extra   string
	}

	var columns []Column
	if err := db.Raw("SHOW COLUMNS FROM inference_histories").Scan(&columns).Error; err != nil {
		fmt.Printf("查询表结构失败: %v\n", err)
		return
	}

	fmt.Println("inference_histories表结构:")
	for _, col := range columns {
		fmt.Printf("字段: %s, 类型: %s, 可空: %s, 键: %s, 默认值: %v, 额外: %s\n",
			col.Field, col.Type, col.Null, col.Key, col.Default, col.Extra)
	}

	// 检查user_audio_models表结构
	var modelColumns []Column
	if err := db.Raw("SHOW COLUMNS FROM user_audio_models").Scan(&modelColumns).Error; err != nil {
		fmt.Printf("查询user_audio_models表结构失败: %v\n", err)
		return
	}

	fmt.Println("\nuser_audio_models表结构:")
	for _, col := range modelColumns {
		fmt.Printf("字段: %s, 类型: %s, 可空: %s, 键: %s, 默认值: %v, 额外: %s\n",
			col.Field, col.Type, col.Null, col.Key, col.Default, col.Extra)
	}
}
