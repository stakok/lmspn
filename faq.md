# FAQ


Your publications acknowledge the use or contribution made by the
software to your research using the following citation:
Wei-Chen Cheng, Stanley Kok, and Hoai Vu Pham (Singapore University of
Technology and Design), Hai Leong Chieu, and Kian Ming A. Chai (DSO
National Labs), "Language Modeling with Sum-Product Networks",
INTERSPEECH, 2014, Singapore.

===

1.  In Figure 2,  the H nodes are used for compression. What are the G nodes for?

    Each G node transforms its input by squaring it, so as to create a
richer representation of the data, which potentially could better
capture the complicated dependency among the input words. Intuitively,
the G nodes are trying to capture the covariance structure between the
input words that has propagated up the H and M layers.


2.  SPN-4’ is initialized using the weights of SPN-3’. What are the PPL
and +KN5 for SPN-3’?

    <table>
    <tr>
        <td></td>
        <td>PPL</td>
        <td>+KN5</td>
    </tr>
    <tr>
        <td>SPN-3':</td>
        <td>101.6</td>
        <td>81.3</td>
    </tr>
    </table>
