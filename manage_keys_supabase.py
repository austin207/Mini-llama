import asyncio
import argparse
from datetime import datetime
from api_server import SupabaseAPIKeyManager

async def main():
    parser = argparse.ArgumentParser(description='Manage Mini-LLaMA API keys with Supabase')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create key
    create_parser = subparsers.add_parser('create', help='Create new API key')
    create_parser.add_argument('--name', required=True, help='Key name')
    create_parser.add_argument('--expires', type=int, help='Expires in days')
    create_parser.add_argument('--rate-limit', type=int, default=100, help='Rate limit per hour')
    
    # List keys
    list_parser = subparsers.add_parser('list', help='List all API keys')
    
    # Stats
    stats_parser = subparsers.add_parser('stats', help='Get key statistics')
    stats_parser.add_argument('--key-id', required=True, help='Key ID')
    
    args = parser.parse_args()
    
    manager = SupabaseAPIKeyManager()
    
    if args.command == 'create':
        result = await manager.generate_api_key(
            name=args.name,
            expires_days=args.expires,
            rate_limit=args.rate_limit
        )
        
        print("✅ API Key Created Successfully!")
        print(f"🔑 Key ID: {result['key_id']}")
        print(f"🔐 API Key: {result['api_key']}")
        print(f"📝 Name: {result['name']}")
        print(f"⏰ Expires: {result['expires_at'] or 'Never'}")
        print(f"🚦 Rate Limit: {result['rate_limit']} requests/hour")
        print(f"🗄️  Stored in: Supabase Database")
        print("\n⚠️  IMPORTANT: Save this API key securely. It won't be shown again!")
        
    elif args.command == 'list':
        keys = await manager.list_api_keys()
        
        print("📋 API Keys (Supabase):")
        print("-" * 100)
        print(f"{'Key ID':<20} {'Name':<20} {'Active':<8} {'Rate Limit':<12} {'Requests':<10} {'Created':<20}")
        print("-" * 100)
        
        for key in keys:
            status = "✅ Yes" if key['is_active'] else "❌ No"
            created = key['created_at'][:19] if key['created_at'] else 'Unknown'
            print(f"{key['key_id']:<20} {key['name']:<20} {status:<8} {key['rate_limit']:<12} {key['total_requests']:<10} {created:<20}")
        
    elif args.command == 'stats':
        stats = await manager.get_usage_stats(args.key_id)
        
        print(f"📊 Statistics for Key: {args.key_id}")
        print("-" * 50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Average Response Time: {stats['average_response_time']:.2f}ms")
        print(f"Total Tokens Generated: {stats['total_tokens_generated']}")
        print(f"Requests Today: {stats['requests_today']}")

if __name__ == "__main__":
    asyncio.run(main())
