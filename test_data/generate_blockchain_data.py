#!/usr/bin/env python3
import json
import random
import datetime
from typing import List, Dict, Any

# 상수 정의
NUM_ADDRESSES = 150  # 주소 수 증가
MIN_EVENTS = 10
MAX_EVENTS = 150  # 최대 이벤트 수 증가

# 계약 타입과 관련 계약 및 이벤트 정의
CONTRACT_TYPES = {
    "defi": {
        "contracts": [
            {"address": "0xc0a47dFe034B400B47bDaD5FecDa2621de6c4d95", "name": "Uniswap V1"},
            {"address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "name": "Uniswap V2 Router"},
            {"address": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45", "name": "Uniswap V3 Router"},
            {"address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9", "name": "Aave Lending Pool"},
            {"address": "0x3dfd23A6c5E8BbcFc9581d2E864a68feb6a076d3", "name": "Compound"},
            {"address": "0x9D25057e62939D3408406975aD75Ffe834DA4cDd", "name": "Yearn Finance"},
            {"address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "name": "DAI Stablecoin"},
            {"address": "0x2a8e1e676ec238d8A992307B495b45B3fEAa5e86", "name": "Curve Finance"},
            {"address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "name": "Uniswap Token"},
            {"address": "0xDef1CA1fb7FBcDC777520aa7f396b4E015F497aB", "name": "Curve Finance DAO"},
        ],
        "events": [
            "swap", "deposit", "withdraw", "borrow", "repay", "liquidate", "stake", "unstake", 
            "claim", "addLiquidity", "removeLiquidity", "harvest", "flash_loan", "bridge", "wrap", "unwrap"
        ]
    },
    "nft": {
        "contracts": [
            {"address": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D", "name": "Bored Ape Yacht Club"},
            {"address": "0x60E4d786628Fea6478F785A6d7e704777c86a7c6", "name": "Mutant Ape Yacht Club"},
            {"address": "0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB", "name": "CryptoPunks"},
            {"address": "0x059EDD72Cd353dF5106D2B9cC5ab83a52287aC3a", "name": "Azuki"},
            {"address": "0x7Bd29408f11D2bFC23c34f18275bBf23bB716Bc7", "name": "Meebits"},
            {"address": "0x1A92f7381B9F03921564a437210bB9396471050C", "name": "Cool Cats"},
            {"address": "0xED5AF388653567Af2F388E6224dC7C4b3241C544", "name": "Azuki"},
            {"address": "0x8a90CAb2b38dba80c64b7734e58Ee1dB38B8992e", "name": "Doodles"},
            {"address": "0x23581767a106ae21c074b2276D25e5C3e136a68b", "name": "Moonbirds"},
            {"address": "0x5Af0D9827E0c53E4799BB226655A1de152A425a5", "name": "Milady Maker"},
        ],
        "events": [
            "mint", "transfer", "sale", "bid", "ask", "auction_created", "auction_success", "auction_cancelled",
            "offer", "offer_accepted", "stake", "claim_rewards", "reveal", "airdrop_claimed"
        ]
    },
    "dao": {
        "contracts": [
            {"address": "0x0bEF27FEB58e857046d630B2c03dFb7bae567494", "name": "ENS DAO"},
            {"address": "0x408ED6354d4973f66138C91495F2f2FCbd8724C3", "name": "Uniswap Governance"},
            {"address": "0xFca59Cd816aB1eaD66534D82bc21E7515cE441CF", "name": "Compound Governance"},
            {"address": "0xBE8E3e3618f7474F8cB1d074A26afFef007E98FB", "name": "Maker DAO"},
            {"address": "0x43a4F930F2cC35948d3a6dAF4a814752784038Fc", "name": "ApeCoin DAO"},
            {"address": "0x0DE760D44Cf8F2cE5c5B7E88184A010EA6DAb0ac", "name": "Lido DAO"},
            {"address": "0x22f69C74A012c02ae628a0a1f9ABA611e166c17B", "name": "Optimism DAO"},
            {"address": "0xdA2C9C36958c4CFD96b0777d7D8f5F2a738eD343", "name": "Base DAO"},
        ],
        "events": [
            "vote", "proposal_created", "proposal_executed", "proposal_cancelled", "delegate", "comment", 
            "join", "leave", "reward_claimed", "funding_requested", "funding_approved", "funding_denied"
        ]
    },
    "gaming": {
        "contracts": [
            {"address": "0xF5D669627376EBd411E34b98F19C868c8ABA5ADA", "name": "Axie Infinity"},
            {"address": "0x1A2F71468F656E97c2F86541E57189F59951efe7", "name": "Decentraland"},
            {"address": "0xB766039cc307aE875c580398FA4c2dc5A0E7c71f", "name": "The Sandbox"},
            {"address": "0x06012c8cf97BEaD5deAe237070F9587f8E7A266d", "name": "CryptoKitties"},
            {"address": "0x4fee7b061c97c9c496b01dbce9cdb10c02f0a0be", "name": "Gods Unchained"},
            {"address": "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0", "name": "Illuvium"},
            {"address": "0xbaB8D6B89576efaEf7C64D9f2C307F86EC1A7894", "name": "Star Atlas"},
            {"address": "0xC36cF0cFcb5d905B8B513860dB0CFE63F6Cf9F5c", "name": "Guild of Guardians"},
            {"address": "0x888888888889c00c67689029D7856AAC1065EC11", "name": "Axie Infinity Shards"},
        ],
        "events": [
            "game_started", "game_ended", "item_crafted", "item_purchased", "battle_won", "battle_lost",
            "level_up", "quest_completed", "reward_claimed", "character_created", "upgrade", "tournament_joined",
            "tournament_ended", "guild_joined", "guild_created", "achievement_unlocked"
        ]
    },
    "social": {  # 새로운 계약 타입 추가
        "contracts": [
            {"address": "0x3835F96B97Adba75Ef9fc78f9c7e7C29383C26F2", "name": "Lens Protocol"},
            {"address": "0xD1b95E5C48c8E9d2CDf4fA5DFdF826aE7A85C2d3", "name": "Farcaster"},
            {"address": "0x3C03b702Ef469c08d7cDe145C65f47D40De62aED", "name": "CyberConnect"},
            {"address": "0x7F062ed5Fb3359fE520D7F5dBDb59486191Ea94b", "name": "Friend.tech"},
            {"address": "0xBD7e92A4d0D9Cf9911b6c7Bab586Ca60C4D69C2E", "name": "Mask Network"},
        ],
        "events": [
            "post_created", "post_liked", "post_shared", "follow", "unfollow", "comment_added", 
            "profile_created", "profile_updated", "content_monetized", "subscription_created", 
            "content_unlocked", "airdrop_claimed", "message_sent", "group_joined"
        ]
    },
    "undefined": {  # 정의되지 않은 새로운 유형의 계약
        "contracts": [
            {"address": "0xAA19861A3BF89D92Cc5352CCc2Dfee86F6C6CA93", "name": "Unknown Protocol A"},
            {"address": "0xEF1c6E67703c7BD7107eed8303Fbe6EC2554BF6B", "name": "Unknown DApp B"},
            {"address": "0x9a76De3b18243D54fBfF9669ecf605C32F464AF8", "name": "Experimental Contract"},
            {"address": "0x1D53a63fc7c7a9e408bbaF2DBdec93358Ed6eeF3", "name": "Novel DApp"},
            {"address": "0x7Cc655A84762C3a0bF28bbA176dB3777c82A9DCb", "name": "Beta Service"},
        ],
        "events": [
            "unknown_event_1", "unknown_event_2", "unknown_event_3", "unknown_event_4", 
            "experimental_action", "beta_feature_used", "alpha_test", "early_access",
            "custom_action", "undefined_interaction"
        ]
    }
}

# 기존 사용자 유형 정의 (16개 성격 유형)
BASE_USER_TYPES = [
    {"type": "D-T-A", "description": "DeFi 트레이더 (위험 감수)", "weights": {"defi": 0.7, "nft": 0.1, "dao": 0.1, "gaming": 0.1}},
    {"type": "D-T-S", "description": "DeFi 트레이더 (안전 중시)", "weights": {"defi": 0.7, "nft": 0.1, "dao": 0.1, "gaming": 0.1}},
    {"type": "D-T-C", "description": "DeFi 트레이더 (커뮤니티 활발)", "weights": {"defi": 0.6, "nft": 0.1, "dao": 0.2, "gaming": 0.1}},
    {"type": "D-T-I", "description": "DeFi 트레이더 (독립적)", "weights": {"defi": 0.8, "nft": 0.1, "dao": 0.0, "gaming": 0.1}},
    {"type": "D-H-A", "description": "DeFi 홀더 (위험 감수)", "weights": {"defi": 0.7, "nft": 0.1, "dao": 0.1, "gaming": 0.1}},
    {"type": "D-H-S", "description": "DeFi 홀더 (안전 중시)", "weights": {"defi": 0.7, "nft": 0.1, "dao": 0.1, "gaming": 0.1}},
    {"type": "D-H-C", "description": "DeFi 홀더 (커뮤니티 활발)", "weights": {"defi": 0.6, "nft": 0.1, "dao": 0.2, "gaming": 0.1}},
    {"type": "D-H-I", "description": "DeFi 홀더 (독립적)", "weights": {"defi": 0.8, "nft": 0.1, "dao": 0.0, "gaming": 0.1}},
    {"type": "N-T-A", "description": "NFT 트레이더 (위험 감수)", "weights": {"defi": 0.1, "nft": 0.7, "dao": 0.1, "gaming": 0.1}},
    {"type": "N-T-S", "description": "NFT 트레이더 (안전 중시)", "weights": {"defi": 0.1, "nft": 0.7, "dao": 0.1, "gaming": 0.1}},
    {"type": "N-T-C", "description": "NFT 트레이더 (커뮤니티 활발)", "weights": {"defi": 0.1, "nft": 0.6, "dao": 0.2, "gaming": 0.1}},
    {"type": "N-T-I", "description": "NFT 트레이더 (독립적)", "weights": {"defi": 0.1, "nft": 0.8, "dao": 0.0, "gaming": 0.1}},
    {"type": "N-H-A", "description": "NFT 홀더 (위험 감수)", "weights": {"defi": 0.1, "nft": 0.7, "dao": 0.1, "gaming": 0.1}},
    {"type": "N-H-S", "description": "NFT 홀더 (안전 중시)", "weights": {"defi": 0.1, "nft": 0.7, "dao": 0.1, "gaming": 0.1}},
    {"type": "N-H-C", "description": "NFT 홀더 (커뮤니티 활발)", "weights": {"defi": 0.1, "nft": 0.6, "dao": 0.2, "gaming": 0.1}},
    {"type": "N-H-I", "description": "NFT 홀더 (독립적)", "weights": {"defi": 0.1, "nft": 0.8, "dao": 0.0, "gaming": 0.1}},
]

# 추가 사용자 유형 정의
ADDITIONAL_USER_TYPES = [
    # 게임 중심 유형
    {"type": "G-T-C", "description": "게임 트레이더 (커뮤니티 활발)", 
     "weights": {"defi": 0.1, "nft": 0.2, "dao": 0.1, "gaming": 0.5, "social": 0.1}},
    {"type": "G-H-A", "description": "게임 홀더 (위험 감수)", 
     "weights": {"defi": 0.0, "nft": 0.2, "dao": 0.0, "gaming": 0.7, "social": 0.1}},
    
    # 소셜 중심 유형
    {"type": "S-T-C", "description": "소셜 활동가 (트레이더)", 
     "weights": {"defi": 0.1, "nft": 0.2, "dao": 0.1, "gaming": 0.1, "social": 0.5}},
    {"type": "S-H-C", "description": "소셜 활동가 (홀더)", 
     "weights": {"defi": 0.1, "nft": 0.1, "dao": 0.1, "gaming": 0.1, "social": 0.6}},
    
    # 혼합 유형
    {"type": "M-T-A", "description": "다양한 활동가 (트레이더, 위험 감수)", 
     "weights": {"defi": 0.3, "nft": 0.3, "dao": 0.1, "gaming": 0.2, "social": 0.1}},
    {"type": "M-H-S", "description": "다양한 활동가 (홀더, 안전 중시)", 
     "weights": {"defi": 0.25, "nft": 0.25, "dao": 0.2, "gaming": 0.2, "social": 0.1}},
    
    # 미정의/실험적 유형
    {"type": "U-T-A", "description": "미정의 활동가 (트레이더, 위험 감수)", 
     "weights": {"defi": 0.1, "nft": 0.1, "dao": 0.1, "gaming": 0.1, "social": 0.1, "undefined": 0.5}},
    {"type": "U-H-I", "description": "미정의 활동가 (홀더, 독립적)", 
     "weights": {"defi": 0.05, "nft": 0.05, "dao": 0.0, "gaming": 0.0, "social": 0.0, "undefined": 0.9}},
    
    # 완전 무작위 패턴
    {"type": "R-X-X", "description": "무작위 패턴", 
     "weights": "random"}  # 이 값은 generate_random_weights 함수에서 처리됨
]

# 모든 유형 통합
USER_TYPES = BASE_USER_TYPES + ADDITIONAL_USER_TYPES

def generate_random_weights() -> Dict[str, float]:
    """완전히 무작위 가중치 생성"""
    contract_types = list(CONTRACT_TYPES.keys())
    weights = {}
    
    # 무작위 0-1 값 할당
    for contract_type in contract_types:
        weights[contract_type] = random.random()
    
    # 정규화 (합이 1이 되도록)
    total = sum(weights.values())
    if total > 0:  # 0으로 나누기 방지
        for contract_type in contract_types:
            weights[contract_type] /= total
    
    return weights

def generate_ethereum_address() -> str:
    """무작위 이더리움 주소 생성"""
    return "0x" + "".join(random.choice("0123456789abcdef") for _ in range(40))

def generate_timestamp(user_type: Dict, event_index: int, total_events: int) -> str:
    """사용자 유형에 따른 타임스탬프 생성"""
    now = datetime.datetime.now()
    
    # 홀더 유형은 더 긴 기간에 걸쳐 이벤트 분산
    if "H" in user_type["type"]:
        days_ago = random.randint(0, 365)  # 최대 1년
    else:
        days_ago = random.randint(0, 90)   # 최대 3개월
    
    # 트레이더는 더 짧은 간격으로 이벤트 발생
    if "T" in user_type["type"]:
        # 이벤트 간격을 더 빽빽하게 생성
        relative_position = event_index / total_events
        # 최근 활동일수록 더 짧은 간격
        days_ago = int(days_ago * (1 - relative_position * 0.8))
    
    # 타임스탬프 생성
    event_date = now - datetime.timedelta(days=days_ago, 
                                        hours=random.randint(0, 23),
                                        minutes=random.randint(0, 59))
    return event_date.strftime("%Y-%m-%d %H:%M:%S")

def generate_amount() -> float:
    """트랜잭션 금액 생성"""
    # 80%는 작은 금액, 20%는 큰 금액
    if random.random() < 0.8:
        return round(random.uniform(0.001, 5.0), 6)
    else:
        return round(random.uniform(5.0, 100.0), 6)

def generate_event_data(contract_type: str, event_name: str) -> Dict:
    """이벤트별 데이터 생성"""
    if contract_type == "defi":
        if event_name in ["swap", "deposit", "withdraw", "borrow", "repay", "stake", "addLiquidity", "removeLiquidity"]:
            return {"amount": generate_amount(), "token": random.choice(["ETH", "USDC", "DAI", "WBTC", "UNI", "AAVE", "COMP", "MKR", "LINK"])}
        else:
            return {"amount": generate_amount()}
            
    elif contract_type == "nft":
        if event_name == "mint":
            return {"tokenId": random.randint(1, 10000), "price": generate_amount()}
        elif event_name in ["sale", "bid", "ask"]:
            return {"tokenId": random.randint(1, 10000), "price": generate_amount()}
        else:
            return {"tokenId": random.randint(1, 10000)}
            
    elif contract_type == "dao":
        if event_name == "vote":
            return {"proposal": random.randint(1, 50), "support": random.choice([True, False])}
        elif event_name == "proposal_created":
            return {"proposal": random.randint(1, 50), "title": "Proposal " + str(random.randint(1000, 9999))}
        elif event_name == "comment":
            return {"text": "Comment " + str(random.randint(1000, 9999))}
        else:
            return {"proposal": random.randint(1, 50)}
            
    elif contract_type == "gaming":
        if event_name in ["battle_won", "battle_lost"]:
            return {"opponent": generate_ethereum_address(), "reward": generate_amount()}
        elif event_name in ["item_purchased", "item_crafted"]:
            return {"itemId": random.randint(1, 1000), "price": generate_amount()}
        else:
            return {"character": "Character " + str(random.randint(100, 999))}
    
    elif contract_type == "social":
        if event_name in ["post_created", "post_liked", "post_shared"]:
            return {"content_id": random.randint(1000, 9999), "visibility": random.choice(["public", "private", "friends"])}
        elif event_name in ["follow", "unfollow"]:
            return {"target_user": generate_ethereum_address()}
        else:
            return {"profile_id": random.randint(1000, 9999)}
    
    elif contract_type == "undefined":
        # 정의되지 않은 타입은 랜덤 데이터 생성
        data = {}
        if random.random() < 0.5:
            data["value"] = generate_amount()
        if random.random() < 0.3:
            data["param_id"] = random.randint(1, 1000)
        if random.random() < 0.4:
            data["target"] = generate_ethereum_address()
        if random.random() < 0.2:
            data["options"] = random.randint(1, 5)
        return data
    
    return {}

def generate_events_for_user(user_address: str, user_type: Dict) -> List[Dict]:
    """특정 사용자 유형에 맞는 이벤트 생성"""
    # 무작위 패턴인 경우 가중치 생성
    if user_type["weights"] == "random":
        weights = generate_random_weights()
    else:
        weights = user_type["weights"]
    
    # 완전 무작위 유형에 대한 이벤트 수 조정
    if user_type["type"] == "R-X-X":
        num_events = random.randint(MIN_EVENTS, MAX_EVENTS * 2)  # 더 많은 이벤트 가능성
    else:
        num_events = random.randint(MIN_EVENTS, MAX_EVENTS)
    
    events = []
    
    # 트레이더는 더 많은 이벤트
    if "T" in user_type["type"]:
        num_events = int(num_events * 1.5)
        
    # 독립적 사용자는 더 적은 계약과 상호작용
    if "I" in user_type["type"]:
        contract_variety = 0.3  # 적은 계약 다양성
    else:
        contract_variety = 0.7  # 높은 계약 다양성
        
    # 사용자가 상호작용할 계약 선택
    user_contracts = {}
    for contract_type, weight in weights.items():
        if weight > 0 and contract_type in CONTRACT_TYPES:
            all_contracts = CONTRACT_TYPES[contract_type]["contracts"]
            # 계약 다양성에 따라 사용할 계약 수 결정
            num_contracts = max(1, int(len(all_contracts) * contract_variety))
            user_contracts[contract_type] = random.sample(all_contracts, num_contracts)
    
    # 이벤트 생성
    for i in range(num_events):
        # 사용자 유형에 따른 계약 타입 선택
        available_types = [ct for ct, w in weights.items() if w > 0 and ct in CONTRACT_TYPES]
        if not available_types:  # 가능한 계약 타입이 없으면 기본 타입 사용
            available_types = ["defi", "nft"]
            
        contract_type = random.choices(
            available_types,
            weights=[weights[ct] for ct in available_types]
        )[0]
        
        # 해당 계약 타입에서 무작위 계약 선택
        if contract_type not in user_contracts or not user_contracts[contract_type]:
            # 계약이 없는 경우 해당 타입의 모든 계약에서 하나 선택
            contract = random.choice(CONTRACT_TYPES[contract_type]["contracts"])
        else:
            contract = random.choice(user_contracts[contract_type])
        
        # 이벤트 생성
        event_name = random.choice(CONTRACT_TYPES[contract_type]["events"])
        
        # 타임스탬프 생성 (사용자 유형 고려)
        timestamp = generate_timestamp(user_type, i, num_events)
        
        # 이벤트 데이터 생성
        event_data = generate_event_data(contract_type, event_name)
        
        # 전체 이벤트 생성
        event = {
            "time": timestamp,
            "sender": user_address,
            "contract": contract["address"],
            "name": contract["name"],
            "type": contract_type,
            "event": event_name,
            "data": event_data
        }
        
        events.append(event)
    
    # 타임스탬프로 정렬
    events.sort(key=lambda x: x["time"])
    return events

def generate_test_data() -> Dict[str, List[Dict]]:
    """전체 테스트 데이터 생성"""
    data = {}
    
    # 기본 유형 + 추가 유형 뿐만 아니라, 완전 무작위 패턴도 더 많이 생성
    random_count = NUM_ADDRESSES // 10  # 전체의 약 10%를 완전 무작위 패턴으로
    remaining = NUM_ADDRESSES - random_count
    
    # 유형별 주소 분배 계산
    addresses_per_type = remaining // (len(USER_TYPES) - 1)  # R-X-X 유형 제외
    extra = remaining % (len(USER_TYPES) - 1)
    
    # 각 유형별로 주소 생성
    for user_type in USER_TYPES:
        # 이 유형에 할당할 주소 수 결정
        if user_type["type"] == "R-X-X":
            num_addresses = random_count
        else:
            num_addresses = addresses_per_type
            if extra > 0:
                num_addresses += 1
                extra -= 1
            
        # 주소 생성
        for _ in range(num_addresses):
            address = generate_ethereum_address()
            events = generate_events_for_user(address, user_type)
            data[address] = events
            
    return data

def save_test_data(data: Dict[str, List[Dict]], filename: str = "blockchain_test_data.json"):
    """테스트 데이터 저장"""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"테스트 데이터가 {filename}에 저장되었습니다.")
    print(f"총 {len(data)} 개의 주소에 대한 데이터가 생성되었습니다.")
    
    # 이벤트 통계
    total_events = sum(len(events) for events in data.values())
    print(f"총 {total_events} 개의 이벤트가 생성되었습니다.")
    
    # 계약 타입별 통계
    contract_type_counts = {}
    for events in data.values():
        for event in events:
            contract_type = event.get("type", "unknown")
            contract_type_counts[contract_type] = contract_type_counts.get(contract_type, 0) + 1
    
    print("\n계약 타입별 이벤트 수:")
    for contract_type, count in sorted(contract_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {contract_type}: {count} ({count/total_events*100:.1f}%)")
    
    # 각 주소별로 이벤트 수 확인
    min_events = float("inf")
    max_events = 0
    for events in data.values():
        events_count = len(events)
        if events_count < min_events:
            min_events = events_count
        if events_count > max_events:
            max_events = events_count
    
    print(f"\n주소별 최소 이벤트: {min_events}")
    print(f"주소별 최대 이벤트: {max_events}")
    print(f"주소당 평균 이벤트: {total_events/len(data):.1f}")

# 메인 실행 코드
if __name__ == "__main__":
    test_data = generate_test_data()
    save_test_data(test_data)
