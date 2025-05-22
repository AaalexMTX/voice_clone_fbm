package main

import (
	"fmt"

	"voice_clone_fbm/backend/config"
	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/router"
	"voice_clone_fbm/backend/utils"

	log "github.com/sirupsen/logrus"
)

func main() {
	// 初始化日志
	utils.InitLogger()

	// 初始化配置
	if err := config.Init(); err != nil {
		log.Fatalf("配置初始化失败: %v", err)
	}

	// 初始化数据库
	if err := model.InitDB(); err != nil {
		log.Fatalf("数据库初始化失败: %v", err)
	}

	// 初始化路由
	r := router.InitRouter()

	// 启动服务器
	addr := fmt.Sprintf("%s:%d", config.GlobalConfig.Server.Host, config.GlobalConfig.Server.Port)
	log.Infof("服务器启动在 %s", addr)
	if err := r.Run(addr); err != nil {
		log.Fatalf("服务器启动失败: %v", err)
	}
}
