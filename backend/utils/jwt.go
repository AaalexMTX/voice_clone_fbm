package utils

import (
	"time"

	"github.com/golang-jwt/jwt"
)

const SecretKey = "your-secret-key" // 在实际应用中应该从配置文件读取

// GenerateToken 生成JWT token
func GenerateToken(username string) (string, error) {
	claims := jwt.MapClaims{
		"username": username,
		"exp":      time.Now().Add(time.Hour * 24).Unix(), // 24小时过期
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(SecretKey))
}
