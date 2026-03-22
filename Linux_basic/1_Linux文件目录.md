# Linux 文件目录笔记

Linux 的文件系统目录结构比较固定，每个目录都有其特定用途。下面按照分类整理常用目录及说明。

---

## 1️⃣ 系统目录（System Directories）

### /bin
- 存放系统常用的二进制可执行文件。
- 示例：`ls`, `cp`, `mv` 等基本命令。

### /sbin
- 存放系统管理命令。
- 示例：`ifconfig`, `shutdown` 等，普通用户一般不直接使用。

### /lib
- 存放系统库文件。
- 示例：`/lib/libc.so`。

### /lib64
- 64 位系统库文件。
- 示例：`/lib64/libc.so.6`。

### /boot
- 存放启动 Linux 所需的核心文件和引导程序。
- 示例：内核镜像 `vmlinuz`，启动配置文件 `grub.cfg`。

---

## 2️⃣ 用户目录（User Directories）

### /home
- 用户个人目录，每个用户有自己的子目录。
- 示例：`/home/username` 存放用户文档、配置文件等。

### /root
- 系统管理员（root 用户）的主目录。
- 类似 `/home/root`。

### /usr
- 存放用户应用程序和文件。
- 类似 Windows 下的 `Program Files` 目录。

#### /usr/local
- 用于主机额外安装软件，通常是源码编译安装。
- 示例：自编译安装的程序。

### /opt
- 存放额外安装的软件。
- 示例：Oracle 数据库安装目录。
- 默认情况下目录为空。

---

## 3️⃣ 可变目录（Variable/Temporary Directories）

### /var
- 存放经常被修改或不断扩充的文件。
- 示例：日志文件 `/var/log`、缓存 `/var/cache`、邮件 `/var/spool`、数据库文件。

### /tmp
- 存放临时文件。
- 系统和用户都可以使用。

### /run
- 存放系统运行时产生的临时文件。
- 示例：PID 文件、服务状态文件等。

---

## 4️⃣ 设备和挂载点（Device & Mount Points）

### /dev
- device，将硬件设备以文件的形式存储。
- 类似 Windows 的设备管理器。

### /media
- 系统自动挂载设备的目录。
- 示例：U 盘、光驱挂载。

### /mnt
- 用户临时挂载其他文件系统的目录。
- 示例：将外部存储挂载到 `/mnt/myshare`。

---

## 5️⃣ 虚拟系统目录（Virtual/System Info Directories）

### /proc
- 虚拟目录，系统内存的映射。
- 通过访问可以获取系统信息。
- 不是实际存储在磁盘上，内容动态生成。

### /sys
- Linux 2.6 内核新增的 `sysfs` 文件系统。
- 提供系统和内核信息接口。

### /selinux
- SELinux 是 Linux 的安全子系统。
- 功能：控制程序只能访问特定文件。
- 三种工作模式：
  - `Enforcing`：强制执行安全策略
  - `Permissive`：记录违规行为，但不阻止
  - `Disabled`：禁用 SELinux