package main

import (
	"fmt"
	"os"

	"voice_clone_fbm/backend/config"
	"voice_clone_fbm/backend/model"
)

func main() {
	// 初始化配置
	if err := config.Init(); err != nil {
		fmt.Printf("配置初始化失败: %v\n", err)
		os.Exit(1)
	}

	// 初始化数据库连接
	if err := model.InitDB(); err != nil {
		fmt.Printf("数据库初始化失败: %v\n", err)
		os.Exit(1)
	}

	// 获取数据库连接
	db := model.GetDB()
	if db == nil {
		fmt.Println("数据库连接为空")
		os.Exit(1)
	}

	// 获取当前数据库中已存在的表
	var tables []string
	if err := db.Raw("SHOW TABLES").Pluck("Tables_in_"+db.Migrator().CurrentDatabase(), &tables).Error; err != nil {
		fmt.Printf("获取表列表失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("当前数据库表:", tables)

	// 检查推理历史记录表是否存在
	inferenceHistoryExists := false
	for _, table := range tables {
		if table == "inference_histories" {
			inferenceHistoryExists = true
			break
		}
	}

	// 如果表不存在，创建表
	if !inferenceHistoryExists {
		fmt.Println("推理历史记录表不存在，开始创建...")
		if err := db.Migrator().CreateTable(&model.InferenceHistory{}); err != nil {
			fmt.Printf("创建推理历史记录表失败: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("推理历史记录表创建成功")
	} else {
		fmt.Println("推理历史记录表已存在，尝试更新结构...")
		if err := db.AutoMigrate(&model.InferenceHistory{}); err != nil {
			fmt.Printf("更新推理历史记录表结构失败: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("推理历史记录表结构更新成功")
	}

	// 重新获取表列表，检查是否成功创建
	if err := db.Raw("SHOW TABLES").Pluck("Tables_in_"+db.Migrator().CurrentDatabase(), &tables).Error; err != nil {
		fmt.Printf("获取表列表失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("迁移后数据库表:", tables)

	fmt.Println("迁移完成")
}
