package service

import (
	"errors"
	"voice_clone_fbm/backend/model"
	"voice_clone_fbm/backend/utils"

	log "github.com/sirupsen/logrus"
)

// UserRegisterReq 用户注册请求结构
type UserRegisterReq struct {
	Username string
	Password string
	Nickname string
	Email    string
}

// UserLoginReq 用户登录请求结构
type UserLoginReq struct {
	Username string
	Password string
}

// UserUpdateReq 用户更新请求结构
type UserUpdateReq struct {
	Nickname string
	Email    string
}

// UserResponse 用户信息响应结构
type UserResponse struct {
	ID       uint
	Username string
	Nickname string
	Email    string
	Role     int8
}

// Register 用户注册服务
func Register(req *UserRegisterReq) (*UserResponse, error) {
	// 检查用户名是否已存在
	if _, err := model.GetByUsername(req.Username); err == nil {
		log.Errorf("用户名已存在: %v", err)
		return nil, errors.New("用户名已存在")
	}

	// 创建新用户
	user := &model.User{
		Username: req.Username,
		Password: req.Password,
		Nickname: req.Nickname,
		Email:    req.Email,
	}

	if err := user.Create(); err != nil {
		log.Errorf("创建用户失败: %v", err)
		return nil, errors.New("创建用户失败")
	}

	// 返回用户信息
	return &UserResponse{
		ID:       user.ID,
		Username: user.Username,
		Nickname: user.Nickname,
		Email:    user.Email,
	}, nil
}

// Login 用户登录服务
func Login(req *UserLoginReq) (*model.User, string, error) {
	// 获取用户
	user, err := model.GetByUsername(req.Username)
	if err != nil {
		log.Errorf("用户不存在: %v", err)
		return nil, "", errors.New("用户名或密码错误")
	}

	// 验证密码
	if !user.ValidatePassword(req.Password) {
		log.Errorf("密码错误: %v", err)
		return nil, "", errors.New("用户名或密码错误")
	}

	// 生成JWT token
	token, err := utils.GenerateToken(user.Username)
	if err != nil {
		return nil, "", errors.New("生成token失败")
	}

	return user, token, nil
}

// UpdateUser 更新用户信息服务
func UpdateUser(username string, req *UserUpdateReq) (*UserResponse, error) {
	// 获取用户
	user, err := model.GetByUsername(username)
	if err != nil {
		log.Errorf("用户不存在: %v", err)
		return nil, errors.New("用户不存在")
	}

	// 更新用户信息
	user.Nickname = req.Nickname
	user.Email = req.Email

	if err := user.Update(); err != nil {
		log.Errorf("更新用户信息失败: %v", err)
		return nil, errors.New("更新用户信息失败")
	}

	// 返回更新后的用户信息
	return &UserResponse{
		ID:       user.ID,
		Username: user.Username,
		Nickname: user.Nickname,
		Email:    user.Email,
		Role:     user.Role,
	}, nil
}
