#include <iostream>
#include <map>

using namespace std;

void bit_view(unsigned int a)
{
    int st[32];
    int count=0;
    while(count<32)
    {
        st[count++]=a&1;
        a=(a>>1);
    }
    while(count>0)
        cout<<st[--count];
    cout<<endl;
}


void generate_MASK(unsigned int &b,unsigned int &c)
{
    unsigned int a=-1;
    b=a,c=a;
    unsigned int pa=1;
    unsigned int pb=2;
    for(int i=0;i<32;i+=2)
    {
        unsigned int p_a=~(pa<<i);
        unsigned int p_b=~(pb<<i);
        b=b&p_a;
        c=c&p_b;
    }
}


int bit_count_init(unsigned int a)
{
    int count;
    for(count=0;a;count++)
        a&=a-1;
    return count;
}

map<int,char> mp_1;
map<int,char> mp_2;

void generate()
{
    mp_1.clear();
    mp_2.clear();

    int x=1;
    int y=2;
    int l,r;
    unsigned int l_MASK;
    unsigned int r_MASK;
    generate_MASK(l_MASK,r_MASK);
    for(int i=0;i<(1<<16);i++)
    {
        l=i&l_MASK;
        r=i&r_MASK;
        if(l)
            mp_1[l]=bit_count_init(l);
        if(r)
            mp_2[r]=bit_count_init(r);
    }
}

int bit_count(unsigned int a,unsigned int l_MASK,unsigned int r_MASK)
{
    unsigned int b=a&l_MASK,c=a&r_MASK;
    return mp_1[b&0xFFFF]+mp_1[b>>16]+mp_2[c&0xFFFF]+mp_2[c>>16];
}

int main()
{
    unsigned int l_MASK;
    unsigned int r_MASK;
    generate();
    generate_MASK(l_MASK,r_MASK);
    unsigned int a;
    while(cin>>a)
    {
        bit_view(a);
        cout<<bit_count(a,l_MASK,r_MASK)<<endl;
    }
}
