import { Link } from 'react-router-dom'
import { useLabSettings } from '../lib/labSettingsContext'

const steps = [
  {
    n: 1,
    title: 'Initialize identities',
    file: 'SecureChannel.initialize()',
    body: 'Each peer generates an ML-KEM-768 encapsulation keypair and an ML-DSA-65 signing keypair via liboqs (Python oqs).',
  },
  {
    n: 2,
    title: 'Encapsulate a shared secret',
    file: 'establish_session(peer_public_key)',
    body: 'Initiator calls ML-KEM encapsulate on the peer public key, producing a ciphertext and a shared secret.',
  },
  {
    n: 3,
    title: 'Decapsulate on the responder',
    file: 'receive_session(ciphertext)',
    body: 'Responder decapsulates with its secret key. Both sides now hold the same shared secret (when the implementation is run in Python).',
  },
  {
    n: 4,
    title: 'Derive an AES-GCM session',
    file: 'AESGCMCipher(shared_secret)',
    body: 'The shared secret is used as a 32-byte AES-256-GCM key. Encrypt returns a 12-byte nonce plus ciphertext.',
  },
  {
    n: 5,
    title: 'Sign control messages',
    file: 'sign_control_message / verify_control_message',
    body: 'ML-DSA-65 signs orchestration or scheduling control bytes. Verification uses the signer’s DSA public key.',
  },
]

export default function Pqc() {
  const { settings, updateSetting } = useLabSettings()
  return (
    <section>
      <header className="page-heading">
        <div>
        <h1>Post-quantum secure channel</h1>
      <p className="lede">
        Sequence from <span className="mono">crypto/secure_channel.py</span>, <span className="mono">ml_kem.py</span>,{' '}
        <span className="mono">ml_dsa.py</span>, and <span className="mono">aes_gcm.py</span>.
        </p></div>
      </header>

      <div className="panel">
        <div className="panel-head"><div><h2>PQC overhead assumptions</h2><p className="faint">Choose algorithm profiles and byte costs for packet sizing.</p></div><Link className="badge replay" to="/lab">View packet model</Link></div>
        <div className="grid-2">
          <div className="stack">
            <label className="lab-number"><span>KEM algorithm label</span><select className="select-control" value={settings.kemProfile} onChange={(event) => updateSetting('kemProfile', event.target.value)}><option>No KEM overhead</option><option>ML-KEM-512 · assumed</option><option>ML-KEM-768 · assumed</option><option>ML-KEM-1024 · assumed</option><option>Custom KEM overhead</option></select></label>
            <label className="lab-number"><span>Assumed ciphertext size per session (bytes)</span><input type="number" min="0" max="20000" value={settings.kemSessionBytes} onChange={(event) => updateSetting('kemSessionBytes', Math.min(20000, Math.max(0, Number(event.target.value) || 0)))} /><small>Amortized across the ten synthetic packets in the scenario model.</small></label>
          </div>
          <div className="stack">
            <label className="lab-number"><span>Signature algorithm label</span><select className="select-control" value={settings.signatureProfile} onChange={(event) => updateSetting('signatureProfile', event.target.value)}><option>No signature overhead</option><option>ML-DSA-44 · assumed</option><option>ML-DSA-65 · assumed</option><option>ML-DSA-87 · assumed</option><option>Custom signature overhead</option></select></label>
            <label className="lab-number"><span>Assumed signature size per packet (bytes)</span><input type="number" min="0" max="20000" value={settings.signatureBytes} onChange={(event) => updateSetting('signatureBytes', Math.min(20000, Math.max(0, Number(event.target.value) || 0)))} /><small>This assumption applies once to each modeled packet.</small></label>
          </div>
        </div>
        <p className="faint" style={{ marginBottom: 0 }}>Byte costs default to zero; adjust them to include overhead in packet sizing.</p>
      </div>

      <div className="grid-2">
        <div className="panel handshake">
          {steps.map((step) => (
            <div className="step" key={step.n}>
              <div className="step-n">{step.n}</div>
              <div>
                <h3>{step.title}</h3>
                <div className="faint mono">{step.file}</div>
                <p className="muted">{step.body}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="stack">
          <div className="panel">
            <h2>Default algorithms</h2>
            <table>
              <thead>
                <tr>
                  <th>Role</th>
                  <th>Code default</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>KEM</td>
                  <td className="mono">ML-KEM-768</td>
                </tr>
                <tr>
                  <td>Signature</td>
                  <td className="mono">ML-DSA-65</td>
                </tr>
                <tr>
                  <td>Payload cipher</td>
                  <td className="mono">AES-256-GCM</td>
                </tr>
                <tr>
                  <td>Nonce</td>
                  <td>12 random bytes per encrypt</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="panel">
            <h2>Classical counterparts in-tree</h2>
            <p className="muted">
              <span className="mono">crypto/classical_crypto.py</span> implements X25519 key exchange and Ed25519
              signatures via the Python <span className="mono">cryptography</span> package.{' '}
              <span className="mono">benchmark.py</span> / <span className="mono">benchmark_comparison.py</span> print
              timings to stdout; they do not write a results file this dashboard can load.
            </p>
          </div>
          <div className="panel">
            <h2>Implementation follow-up</h2>
            <ul className="muted">
              <li>Install liboqs / python-oqs and record handshake + AES-GCM microseconds to JSON.</li>
              <li>Include ciphertext and signature sizes in the semantic communication cost model.</li>
              <li>Wire PQC session setup into the network event simulator (currently independent).</li>
              <li>
                Empty stubs <span className="mono">crypto/aes.py</span> and <span className="mono">crypto/handshake.py</span>{' '}
                are unused; live path is <span className="mono">secure_channel.py</span>.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
