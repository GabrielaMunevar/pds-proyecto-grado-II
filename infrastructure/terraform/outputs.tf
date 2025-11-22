# ============================================================================
# Outputs
# ============================================================================

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.pls_api.repository_url
}

output "ecr_repository_name" {
  description = "Name of the ECR repository"
  value       = aws_ecr_repository.pls_api.name
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

output "api_url" {
  description = "URL of the API"
  value       = "http://${aws_lb.main.dns_name}"
}

output "api_docs_url" {
  description = "URL of the API documentation"
  value       = "http://${aws_lb.main.dns_name}/docs"
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.api.name
}

output "dvc_s3_bucket" {
  description = "S3 bucket name for DVC (where models are stored)"
  value       = var.dvc_s3_bucket
}

output "dvc_s3_path" {
  description = "Full S3 path for DVC store"
  value       = "s3://${var.dvc_s3_bucket}/${var.dvc_s3_prefix}"
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.ecs.name
}

