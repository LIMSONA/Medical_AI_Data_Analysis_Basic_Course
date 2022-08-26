def main():
    sensitivity = float(input())
    prior_prob = float(input())
    false_alarm = float(input())

    print("%.2lf%%" % (100 * mammogram_test(sensitivity, prior_prob, false_alarm)))

def mammogram_test(sensitivity, prior_prob, false_alarm):
    p_a1_b1 = sensitivity # p(A = 1 | B = 1) 유방암 보유자를 대상으로 검사 결과가 양성으로 표시될 확률
    
    p_b1 = prior_prob    # p(B = 1) 총 인구를 기준으로 유방암을 가지고 있을 사전 확률(prior probability)
    
    p_b0 = 1-p_b1    # p(B = 0)

    p_a1_b0 = false_alarm # p(A = 1|B = 0) 실제로는 암을 갖고 있지 않지만 유방암이라고 진단될 확률

    p_a1 = p_a1_b0*p_b0 + p_a1_b1*p_b1    # p(A = 1)

    p_b1_a1 = p_a1_b1*p_b1 / p_a1 # p(B = 1|A = 1)

    return p_b1_a1

if __name__ == "__main__":
    main()
