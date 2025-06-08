from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import yaml
import json
import hashlib
import secrets
import time
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pydantic import BaseModel
import threading
from pathlib import Path
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Import your inference engine
from inference import LlamaInference

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Validate environment variables
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")
if not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_SERVICE_KEY environment variable is required")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False

class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    generation_time: float
    tokens_per_second: float
    model_info: Dict

class APIKeyCreate(BaseModel):
    name: str
    expires_days: Optional[int] = None
    rate_limit: Optional[int] = 100

class APIKeyResponse(BaseModel):
    key_id: str
    api_key: str
    name: str
    expires_at: Optional[str]
    rate_limit: int

class UsageStats(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    total_tokens_generated: int
    requests_today: int

# Supabase API Key Manager
class SupabaseAPIKeyManager:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    
    async def generate_api_key(self, name: str, expires_days: Optional[int] = None, rate_limit: int = 100) -> Dict:
        """Generate a new API key and store in Supabase"""
        key_id = f"llama_{secrets.token_hex(8)}"
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        expires_at = None
        if expires_days:
            expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()
        
        try:
            result = self.supabase.table("api_keys").insert({
                "key_id": key_id,
                "key_hash": key_hash,
                "name": name,
                "expires_at": expires_at,
                "rate_limit": rate_limit,
                "is_active": True,
                "total_requests": 0
            }).execute()
            
            if result.data:
                return {
                    "key_id": key_id,
                    "api_key": api_key,
                    "name": name,
                    "expires_at": expires_at,
                    "rate_limit": rate_limit
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to create API key")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key against Supabase database"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        try:
            result = self.supabase.table("api_keys").select(
                "key_id, name, expires_at, is_active, rate_limit, total_requests"
            ).eq("key_hash", key_hash).eq("is_active", True).execute()
            
            if not result.data:
                return None
            
            key_data = result.data[0]
            
            # Check expiration
            if key_data["expires_at"]:
                expires_dt = datetime.fromisoformat(key_data["expires_at"].replace('Z', '+00:00'))
                if datetime.now() > expires_dt.replace(tzinfo=None):
                    return None
            
            return {
                "key_id": key_data["key_id"],
                "name": key_data["name"],
                "rate_limit": key_data["rate_limit"],
                "total_requests": key_data["total_requests"]
            }
            
        except Exception as e:
            print(f"Database error during validation: {e}")
            return None
    
    async def check_rate_limit(self, key_id: str) -> bool:
        """Check if key is within rate limits (requests per hour)"""
        try:
            # Get current rate limit
            key_result = self.supabase.table("api_keys").select("rate_limit").eq("key_id", key_id).execute()
            
            if not key_result.data:
                return False
            
            rate_limit = key_result.data[0]["rate_limit"]
            
            # Count requests in last hour
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
            
            usage_result = self.supabase.table("api_usage").select(
                "id", count="exact"
            ).eq("key_id", key_id).gte("request_time", one_hour_ago).execute()
            
            request_count = usage_result.count or 0
            
            return request_count < rate_limit
            
        except Exception as e:
            print(f"Rate limit check error: {e}")
            return False
    
    async def log_usage(self, key_id: str, endpoint: str, response_time_ms: int, 
                       tokens_generated: int, success: bool, ip_address: str, user_agent: str):
        """Log API usage to Supabase"""
        try:
            # Insert usage record
            self.supabase.table("api_usage").insert({
                "key_id": key_id,
                "endpoint": endpoint,
                "response_time_ms": response_time_ms,
                "tokens_generated": tokens_generated,
                "success": success,
                "ip_address": ip_address,
                "user_agent": user_agent
            }).execute()
            
            # Update total requests count
            self.supabase.table("api_keys").update({
                "total_requests": self.supabase.table("api_keys").select("total_requests").eq("key_id", key_id).execute().data[0]["total_requests"] + 1,
                "last_used": datetime.now().isoformat()
            }).eq("key_id", key_id).execute()
            
        except Exception as e:
            print(f"Usage logging error: {e}")

    async def get_usage_stats(self, key_id: str) -> Dict:
        """Get usage statistics for a specific API key"""
        try:
            # Total requests
            total_result = self.supabase.table("api_usage").select(
                "id", count="exact"
            ).eq("key_id", key_id).execute()
            total_requests = total_result.count or 0
            
            # Successful requests
            success_result = self.supabase.table("api_usage").select(
                "id", count="exact"
            ).eq("key_id", key_id).eq("success", True).execute()
            successful_requests = success_result.count or 0
            
            # Average response time and total tokens
            stats_result = self.supabase.table("api_usage").select(
                "response_time_ms, tokens_generated"
            ).eq("key_id", key_id).eq("success", True).execute()
            
            if stats_result.data:
                response_times = [r["response_time_ms"] for r in stats_result.data if r["response_time_ms"]]
                tokens = [r["tokens_generated"] for r in stats_result.data if r["tokens_generated"]]
                
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                total_tokens = sum(tokens) if tokens else 0
            else:
                avg_response_time = 0
                total_tokens = 0
            
            # Requests today
            today = datetime.now().date().isoformat()
            today_result = self.supabase.table("api_usage").select(
                "id", count="exact"
            ).eq("key_id", key_id).gte("request_time", today).execute()
            requests_today = today_result.count or 0
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": total_requests - successful_requests,
                "average_response_time": avg_response_time,
                "total_tokens_generated": total_tokens,
                "requests_today": requests_today
            }
            
        except Exception as e:
            print(f"Stats retrieval error: {e}")
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0,
                "total_tokens_generated": 0,
                "requests_today": 0
            }

    async def list_api_keys(self) -> List[Dict]:
        """List all API keys (admin function)"""
        try:
            result = self.supabase.table("api_keys").select(
                "key_id, name, created_at, expires_at, is_active, rate_limit, total_requests"
            ).execute()
            
            return result.data or []
            
        except Exception as e:
            print(f"List keys error: {e}")
            return []

# Initialize FastAPI app
app = FastAPI(
    title="Mini-LLaMA API with Supabase",
    description="Custom API for Mini-LLaMA language model with Supabase-powered API key authentication",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
api_key_manager = SupabaseAPIKeyManager()
inference_engine = None

# API Key header authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    """Dependency to validate API key"""
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include X-API-Key header."
        )
    
    key_info = await api_key_manager.validate_api_key(api_key)
    if key_info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key"
        )
    
    # Check rate limit
    if not await api_key_manager.check_rate_limit(key_info["key_id"]):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )
    
    return key_info

@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    global inference_engine
    print("🚀 Initializing Mini-LLaMA inference engine...")
    print(f"🔗 Connected to Supabase: {SUPABASE_URL}")
    inference_engine = LlamaInference(inference_config_path="configs/inference.yaml")
    print("✅ API server ready!")

# API Endpoints

@app.get("/")
async def root():
    """API status and information"""
    return {
        "message": "Mini-LLaMA API Server with Supabase",
        "version": "1.0.0",
        "status": "online",
        "database": "Supabase",
        "endpoints": {
            "generate": "/v1/generate",
            "stream": "/v1/stream", 
            "health": "/health",
            "create_key": "/admin/create-key",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "database": "Supabase Connected",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    key_info: Dict = Depends(get_api_key),
    client_request: Request = None,
    background_tasks: BackgroundTasks = None
):
    """Generate text using Mini-LLaMA model"""
    start_time = time.time()
    
    try:
        # Generate text
        result = inference_engine.generate_with_timing(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            verbose=False
        )
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Log usage in background
        background_tasks.add_task(
            api_key_manager.log_usage,
            key_info["key_id"],
            "/v1/generate",
            response_time_ms,
            result["generated_tokens"],
            True,
            client_request.client.host,
            client_request.headers.get("user-agent", "")
        )
        
        return GenerateResponse(
            text=result["text"],
            prompt_tokens=result["prompt_tokens"],
            generated_tokens=result["generated_tokens"],
            total_tokens=result["prompt_tokens"] + result["generated_tokens"],
            generation_time=result["generation_time"],
            tokens_per_second=result["generation_tps"],
            model_info={
                "model_name": "Mini-LLaMA",
                "parameters": "58M",
                "context_length": 1024,
                "backend": "Supabase"
            }
        )
        
    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Log failed usage
        background_tasks.add_task(
            api_key_manager.log_usage,
            key_info["key_id"],
            "/v1/generate",
            response_time_ms,
            0,
            False,
            client_request.client.host,
            client_request.headers.get("user-agent", "")
        )
        
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/v1/stream")
async def stream_generate(
    request: GenerateRequest,
    key_info: Dict = Depends(get_api_key)
):
    """Stream text generation"""
    async def generate_stream():
        try:
            result = inference_engine.generate_with_timing(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                verbose=False
            )
            
            # Simulate streaming
            words = result["text"].split()
            for i, word in enumerate(words):
                chunk = {
                    "text": word + " ",
                    "is_complete": i == len(words) - 1,
                    "tokens_generated": i + 1
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)
                
        except Exception as e:
            error_chunk = {"error": str(e)}
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/stats")
async def get_usage_stats(key_info: Dict = Depends(get_api_key)):
    """Get usage statistics for the API key"""
    stats = await api_key_manager.get_usage_stats(key_info["key_id"])
    return UsageStats(**stats)

# Admin endpoints
@app.post("/admin/create-key", response_model=APIKeyResponse)
async def create_api_key(request: APIKeyCreate):
    """Create a new API key (admin endpoint)"""
    result = await api_key_manager.generate_api_key(
        name=request.name,
        expires_days=request.expires_days,
        rate_limit=request.rate_limit
    )
    
    return APIKeyResponse(**result)

@app.get("/admin/keys")
async def list_api_keys():
    """List all API keys (admin endpoint)"""
    keys = await api_key_manager.list_api_keys()
    return {"api_keys": keys}

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
