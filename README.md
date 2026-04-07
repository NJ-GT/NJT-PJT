# NJT-PJT

## 프로젝트 소개
> 프로젝트 한 줄 설명을 여기에 작성하세요.

- 개발 기간: 1달
- 개발 인원: 4명

---

## 팀원 소개

| 이름 | 역할 | GitHub |
|------|------|--------|
|  남기혁|  |  |
|  이윤원|  |  |
|  김현수|  |  |
|  이대한|  |  eogks1235-byte  |
---

## 기술 스택

- **언어**: 
- **프레임워크**: 
- **데이터베이스**: 
- **배포**: 

---

## 폴더 구조

```
NJT-PJT/
├── 
├── 
└── 
```

---

## 개발 환경 설정

### 요구 사항
- 

### 설치 및 실행
```bash
# 저장소 클론
git clone https://github.com/[저장소 주소].git


```

### 환경변수 설정
`.env` 파일을 루트 디렉토리에 생성 후 아래 내용을 작성하세요.
```
# 예시
DB_HOST=localhost
DB_PORT=3306
```

---

## 코드 작성 규칙

### 한글 주석 필수
모든 함수, 클래스, 주요 로직에 **한글 주석**을 작성합니다.
함수의 역할, 매개변수, 반환값을 명확히 설명해야 합니다.

```python
# Python 예시
def calculate_total(price, quantity):
    """
    총 금액을 계산하는 함수
    :param price: 단가 (int)
    :param quantity: 수량 (int)
    :return: 총 금액 (int)
    """
    return price * quantity
```

## GitHub 사용법

### 브랜치 전략
- `main`: 최종 배포 브랜치 (직접 push 금지)
- `develop`: 개발 통합 브랜치
- `feature/기능명`: 기능 개발 브랜치

### 브랜치 만들기

```bash
# 1. 최신 상태로 업데이트
git pull origin develop

# 2. 새 브랜치 생성 및 이동
git switch feature/name

### 코드 작성 후 커밋 & 푸시

```bash
# 1. 변경된 파일 확인
git status

# 2. 파일 스테이징 (전체 추가)
git add .

# 또는 특정 파일만 추가
git add 파일명.py

# 3. 커밋 (메시지 필수 작성)
git commit -m "feat: 로그인 기능 구현"

# 4. 원격 저장소에 푸시
git push origin 해당브랜치
```

### 작업 흐름 요약

```
1. git pull origin 해당 브랜치         ← 최신 코드 받기
2. git branch feature/name  ← 내 브랜치 만들기
3. 코드 작성
4. git add .                       ← 변경사항 추가
5. git commit -m "커밋 메시지"     ← 커밋
6. git push origin feature/name  ← 푸시
7. GitHub에서 Pull Request 생성    ← 팀원 리뷰 후 develop에 합치기
```

### 자주 쓰는 명령어 정리

| 명령어 | 설명 |
|--------|------|
| `git status` | 변경된 파일 목록 확인 |
| `git branch` | 현재 브랜치 목록 확인 |
| `git checkout 브랜치명` | 브랜치 이동 |
| `git pull origin develop` | develop 브랜치 최신 코드 받기 |
| `git log --oneline` | 커밋 히스토리 간단히 보기 |

---

## 참고 자료

- 
