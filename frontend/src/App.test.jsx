import { render, screen, fireEvent } from '@testing-library/react';
import App from './App.jsx';

describe('App', () => {
  it('renders the main UI', () => {
    render(<App />);
    expect(screen.getByText(/Video Translator UI/i)).toBeInTheDocument();
    expect(screen.getByText(/Start Processing/i)).toBeInTheDocument();
  });

  it('disables Start Processing when no files', () => {
    render(<App />);
    const btn = screen.getByText(/Start Processing/i);
    expect(btn).toBeDisabled();
  });

  it('enables Start Processing when files are added', () => {
    render(<App />);
    const input = screen.getByLabelText(/browse/i).querySelector('input[type="file"]');
    // Simulate file input
    const file = new File(['dummy'], 'test.mp4', { type: 'video/mp4' });
    fireEvent.change(input, { target: { files: [file] } });
    const btn = screen.getByText(/Start Processing/i);
    expect(btn).not.toBeDisabled();
  });
}); 