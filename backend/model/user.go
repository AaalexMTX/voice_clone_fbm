package model

import (
	"errors"
	"strings"
	"voice_clone_fbm/backend/utils"

	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

// User 用户模型
type User struct {
	UID        string `gorm:"type:char(32);not null;unique;index" json:"uid"`   // 用户唯一标识
	Username   string `gorm:"type:varchar(50);not null;unique" json:"username"` // 用户名
	Password   string `gorm:"type:varchar(255);not null" json:"-"`              // 密码
	Nickname   string `gorm:"type:varchar(50)" json:"nickname"`                 // 昵称
	Email      string `gorm:"type:varchar(100)" json:"email"`                   // 邮箱
	Role       int8   `gorm:"type:tinyint;default:1" json:"role"`               // 用户角色(1:user, 2:admin)
	gorm.Model        // ID, CreatedAt, UpdatedAt, DeletedAt
}

// BeforeCreate 创建用户前生成UID
func (u *User) BeforeCreate(tx *gorm.DB) error {
	// 生成32位的UUID作为UID
	u.UID = GenerateUID()
	return nil
}

// GenerateUID 生成用户唯一标识
func GenerateUID() string {
	return strings.ReplaceAll(uuid.New().String(), "-", "")
}

// Create 创建用户
func (u *User) Create() error {
	if err := u.HashPassword(); err != nil {
		return err
	}
	if err := DB.Create(u).Error; err != nil {
		return err
	}
	// 创建用户数据目录
	return utils.InitUserDataDir(u.Username)
}

// GetByUsername 通过用户名获取用户
func GetByUsername(username string) (*User, error) {
	var user User
	err := DB.Where("username = ?", username).First(&user).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, errors.New("用户不存在")
		}
		return nil, err
	}
	return &user, nil
}

// GetByUID 通过UID获取用户
func GetByUID(uid string) (*User, error) {
	var user User
	err := DB.Where("uid = ?", uid).First(&user).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, errors.New("用户不存在")
		}
		return nil, err
	}
	return &user, nil
}

// Update 更新用户信息
func (u *User) Update() error {
	return DB.Save(u).Error
}

// Delete 删除用户
func (u *User) Delete() error {
	return DB.Delete(u).Error
}

// ValidatePassword 验证密码
func (u *User) ValidatePassword(password string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(u.Password), []byte(password))
	return err == nil
}

// HashPassword 加密密码
func (u *User) HashPassword() error {
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(u.Password), bcrypt.DefaultCost)
	if err != nil {
		return err
	}
	u.Password = string(hashedPassword)
	return nil
}

// ChangePassword 修改密码
func (u *User) ChangePassword(newPassword string) error {
	u.Password = newPassword
	if err := u.HashPassword(); err != nil {
		return err
	}
	return u.Update()
}

// IsAdmin 检查用户是否为管理员
func (u *User) IsAdmin() bool {
	return u.Role >= 2 // RoleAdmin
}

// IsSuperAdmin 检查用户是否为超级管理员
func (u *User) IsSuperAdmin() bool {
	return u.Role == 3 // RoleSuper
}
