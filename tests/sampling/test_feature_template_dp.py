from dataclasses import replace
import numpy as np
import pytest
from tri.domain.music import MusicSpec,CountRule
from tri.inference.exact import Budget
from tri.sampling.research_methods import _approximate_template
from tri.errors import BudgetExceeded


@pytest.mark.parametrize('pooled',[False,True])
def test_count_dp_preserves_original_approximate_template_distribution(pooled):
    spec=MusicSpec(length=4,pitches=(60,),equal_onsets=((0,2),(1,3)),
                   onset_counts=(CountRule((0,1),1),CountRule((2,3),1)))
    probs=np.array([.2,.7,.8,.1]);q=np.zeros((4,130))
    q[:,:2]=(1-probs[:,None])/2;q[:,2:]=probs[:,None]/128;q=np.log(q)
    a=probs[0]*(1-probs[1]);b=(1-probs[0])*probs[1]
    if pooled:
        a*=probs[2]*(1-probs[3]);b*=(1-probs[2])*probs[3]
    rng=np.random.default_rng(18)
    draws=[_approximate_template(spec,q,pooled,rng,Budget()) for _ in range(1200)]
    assert all(d[0]+d[1]==1 and d[0]==d[2] and d[1]==d[3] for d in draws)
    assert np.mean([d[0] for d in draws])==pytest.approx(a/(a+b),abs=.035)


def test_two_bar_template_count_dp_scales_and_keeps_observed_flags():
    spec=MusicSpec(length=64,pitches=(60,),observed={0:62},equal_onsets=tuple((i,i+32) for i in range(32)),
                   onset_counts=(CountRule(tuple(range(32)),8),CountRule(tuple(range(32,64)),8)))
    q=np.full((64,130),-np.log(130))
    flags=_approximate_template(spec,q,True,np.random.default_rng(4),Budget(max_factor_entries=1000))
    assert flags[0]==flags[32]==1 and sum(flags[i] for i in range(32))==8
    with pytest.raises(BudgetExceeded):
        _approximate_template(spec,q,True,np.random.default_rng(4),Budget(max_factor_entries=1))
