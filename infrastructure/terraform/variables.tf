# ============================================================================
# Variables para Terraform
# ============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name (used for resource naming)"
  type        = string
  default     = "medical-pls-api"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets (increases cost)"
  type        = bool
  default     = false
}

variable "dvc_s3_bucket" {
  description = "S3 bucket name for DVC (where models are stored)"
  type        = string
  default     = "pds-pls-data-prod"
}

variable "dvc_s3_prefix" {
  description = "S3 prefix/path for DVC store"
  type        = string
  default     = "dvcstore"
}

variable "task_cpu" {
  description = "CPU units for ECS task (256, 512, 1024, 2048, 4096)"
  type        = number
  default     = 2048
}

variable "task_memory" {
  description = "Memory for ECS task in MB (512, 1024, 2048, 4096, 8192, 16384)"
  type        = number
  default     = 4096
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 1
}

variable "min_capacity" {
  description = "Minimum number of tasks for auto scaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of tasks for auto scaling"
  type        = number
  default     = 5
}

variable "cpu_target_value" {
  description = "Target CPU utilization percentage for auto scaling"
  type        = number
  default     = 70.0
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "log_level" {
  description = "Log level for application"
  type        = string
  default     = "INFO"
}

variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights"
  type        = bool
  default     = true
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for ALB"
  type        = bool
  default     = false
}

# ============================================================================
# API Configuration Variables
# ============================================================================

variable "cors_origins" {
  description = "CORS allowed origins (comma-separated, or * for all)"
  type        = string
  default     = "*"
}

variable "max_length" {
  description = "Default maximum output length in tokens"
  type        = number
  default     = 256
}

variable "num_beams" {
  description = "Default number of beams for beam search"
  type        = number
  default     = 4
}

variable "max_input_length" {
  description = "Maximum input tokens per chunk"
  type        = number
  default     = 512
}

variable "chunk_size" {
  description = "Chunk size for text splitting in tokens"
  type        = number
  default     = 400
}

variable "chunk_overlap" {
  description = "Overlap between chunks in tokens"
  type        = number
  default     = 50
}

variable "debug" {
  description = "Enable debug mode (not recommended for production)"
  type        = bool
  default     = false
}

variable "additional_env_vars" {
  description = "Additional environment variables to pass to the container (list of {name, value} objects)"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}

